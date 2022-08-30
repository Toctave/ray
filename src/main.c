#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>
#include <threads.h>
#include <time.h>
#include <unistd.h>

#include "base_types.h"
#include "simd.h"
#include "stb_image_write.h"

#define FETCH_ADD(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)

#if 1

#define ASSERT(expr)                                                                                                   \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, "%s:%d: Assertion %s failed\n", __FILE__, __LINE__, #expr);                                \
            raise(SIGTRAP);                                                                                            \
        }                                                                                                              \
    } while (0)

#else

#define ASSERT(expr)

#endif

#define ASSERT_ZERO(a, tol) ASSERT(fabs(a) < tol)

#define ASSERT_EQ(a, b, tol) ASSERT(fabs(a - b) < tol * fabs(b))

#define ASSERT_V3_NORMALIZED(v, tol) ASSERT_EQ(v3_norm(v), 1.0f, tol)

typedef struct {
    v3 emission;
    v3 diffuse;
} Material;

typedef struct {
    v3 center;
    float radius;
    u32 material_index;
} Sphere;

typedef struct {
    u32 sphere_count;
    Sphere* spheres;
    Material* materials;

    u64 traced_ray_count;
} World;

typedef struct {
    u32 width;
    u32 height;

    u32 stride;
    u8 bytes_per_pixel;

    u8* pixels;
} Image;

typedef struct {
    w_v3 point;
    w_v3 tangent1; // in the plane of the outgoing ray
    w_v3 tangent2; // (t1, t2, normal) is right-handed
    w_v3 normal;
    w_float t;
    w_u32 material_index;
} Intersect;

typedef struct {
    Image image;
    World* world;

    u32 x_min;
    u32 x_max;
    u32 y_min;
    u32 y_max;
} WorkItem;

typedef struct {
    WorkItem* items;
    u32 item_count;
    u32 next_work_item;
} WorkQueue;

v3 v3_sub(v3 u, v3 v)
{
    return (v3){u.x - v.x, u.y - v.y, u.z - v.z};
}

v3 v3_add(v3 u, v3 v)
{
    return (v3){u.x + v.x, u.y + v.y, u.z + v.z};
}

v3 v3_scale(float s, v3 u)
{
    return (v3){u.x * s, u.y * s, u.z * s};
}

v3 v3_div(v3 u, float s)
{
    return (v3){u.x / s, u.y / s, u.z / s};
}

float v3_dot(v3 u, v3 v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}

float v3_norm_squared(v3 u)
{
    return v3_dot(u, u);
}

float v3_norm(v3 u)
{
    return sqrt(v3_norm_squared(u));
}

v3 v3_normalized(v3 u)
{
    return v3_div(u, v3_norm(u));
}

v3 v3_cross(v3 u, v3 v)
{
    return (v3){
        u.y * v.z - u.z * v.y,
        u.z * v.x - u.x * v.z,
        u.x * v.y - u.y * v.x,
    };
}

v3 v3_any_orthogonal(v3 u)
{
    u32 best = 0;
    for (u32 i = 1; i < 3; i++) {
        if (u.co[i] > u.co[best]) {
            best = i;
        }
    }

    v3 result;
    result.co[best] = 0.0f;
    result.co[(best + 1) % 3] = u.co[(best + 2) % 3];
    result.co[(best + 2) % 3] = -u.co[(best + 1) % 3];

    ASSERT_ZERO(v3_dot(result, u), 1.0e-6f);

    return result;
}

v3 v3_in_basis(v3 coords, v3 x, v3 y, v3 z)
{
    return v3_add(v3_add(v3_scale(coords.x, x), v3_scale(coords.y, y)), v3_scale(coords.z, z));
}

v3 v3_hadamard(v3 u, v3 v)
{
    return (v3){u.x * v.x, u.y * v.y, u.z * v.z};
}

w_u32 find_intersect(const World* world, w_v3 orig, w_v3 dir, Intersect* intersect)
{
    w_u32 res = 0;
    for (u32 i = 0; i < world->sphere_count; i++) {
        const Sphere* sphere = &world->spheres[i];

        w_v3 center = w_v3_broadcast(sphere->center);
        w_u32 mat_index = w_u32_broadcast(sphere->material_index);
        w_float radius_squared = w_float_broadcast(sphere->radius * sphere->radius);

        // normSq(o + t * d - c)  = r^2
        // normSq(o - c) + t^2 * normSq(d) + 2.0f * (o - c) . d * t - r^2 = 0

        w_v3 co = w_v3_sub(orig, center);

        w_float b = w_float_mul(w_float_broadcast(2.0f), w_v3_dot(co, dir));
        w_float c = w_float_sub(w_v3_dot(co, co), radius_squared);

        w_float disc = b * b - 4 * c;

        w_float sqrt_disc = sqrt(disc);
        w_float t = (-b - sqrt_disc) / 2.0f;

        w_u32 assign_mask = (disc >= 0.0f) & (t > 0.0f) & (t < intersect->t);
        if (assign_mask) {
            assign_mask = 0xFFFFFFFF;
        }

        w_v3 hit_point = w_v3_add(orig, w_v3_scale(t, dir));
        w_v3 normal = w_v3_normalized(w_v3_sub(hit_point, center));

        /* if (assign_mask) { */
        /* printf("normal : %f %f %f\n", normal.x, normal.y, normal.z); */
        /* } */

        w_float_conditional_assign(assign_mask, &intersect->t, t);
        w_v3_conditional_assign(assign_mask, &intersect->normal, normal);
        w_v3_conditional_assign(assign_mask, &intersect->point, hit_point);
        w_u32_conditional_assign(assign_mask, &intersect->material_index, mat_index);

        res = w_u32_or(res, assign_mask);
    }

    intersect->tangent2 =
        w_v3_normalized(w_v3_cross(dir, intersect->normal)); // TODO(octave) : fallback for orthogonal hits
    intersect->tangent1 = w_v3_cross(intersect->tangent2, intersect->normal);


    /* ASSERT_V3_NORMALIZED(intersect->normal, 1.0e-3f); */
    /* ASSERT_V3_NORMALIZED(intersect->tangent2, 1.0e-3f); */
    /* ASSERT_V3_NORMALIZED(intersect->tangent1, 1.0e-3f); */

    return res;
}

w_u32 random_u32(w_u32* state)
{
    *state = 1103515245 * (*state) + 12345;

    return *state;
}

w_float random_float(w_u32* state)
{
    return (float)random_u32(state) / (float)UINT32_MAX;
}

void random_uniform_circle(w_u32* rng, w_float* x_out, w_float* y_out)
{
#if 0
    w_float x, y;
    do {
        x = 2.0f * random_float(rng) - 1.0f;
        y = 2.0f * random_float(rng) - 1.0f;
    } while (x * x + y * y >= 1.0f);

    *x_out = x;
    *y_out = y;
#else
    w_float r = sqrt(random_float(rng));
    w_float theta = 2.0f * M_PI * random_float(rng);

    *x_out = r * cos(theta);
    *y_out = r * sin(theta);
#endif
}

w_v3 random_cosine_weighted(w_u32* rng)
{
    w_v3 res;
    random_uniform_circle(rng, &res.x, &res.y);
    w_float norm_2d_squared = w_float_add(w_float_mul(res.x, res.x), w_float_mul(res.y, res.y));
    res.z = w_sqrt(w_float_sub(w_float_broadcast(1.0f), norm_2d_squared));

    return res;
}

v3 trace_ray(const World* world, u32* rng, w_v3 orig, w_v3 dir, u32 max_depth, u32* bounces)
{
    w_v3 result = {};
    w_v3 attenuation = w_v3_broadcast((v3){1.0f, 1.0f, 1.0f});
    w_u32 w_bounces = w_u32_broadcast(0);

    w_u32 ray_alive_mask = w_u32_broadcast(0xFFFFFFFF);

    for (u32 depth = 0; depth < max_depth; depth++) {
        Intersect itx = {};
        itx.t = w_float_broadcast(INFINITY);

        w_u32 bounced_mask = find_intersect(world, orig, dir, &itx);

        // TODO(octave) : actual n-way load
        const Material* mat = &world->materials[itx.material_index];
        w_v3 emission = w_v3_broadcast(mat->emission);
        w_v3 diffuse = w_v3_broadcast(mat->diffuse);
        // --------

        emission = w_v3_masked(ray_alive_mask, emission);

        result = w_v3_add(result, w_v3_hadamard(attenuation, emission));
        ray_alive_mask = w_u32_and(ray_alive_mask, bounced_mask);

        w_bounces = w_u32_add(w_bounces, w_u32_and(ray_alive_mask, w_u32_broadcast(1)));

        // no cosine term : embedded in cosine-weighted sampling
        attenuation = w_v3_hadamard(attenuation, diffuse);

        // compute bounce ray
        orig = itx.point;
        dir = w_v3_in_basis(random_cosine_weighted(rng), itx.tangent1, itx.tangent2, itx.normal);

        // bail out if the whole lane is dead
        if (w_mask_is_zeroed(ray_alive_mask)) {
            break;
        }
    }

    *bounces = w_u32_horizontal_add(w_bounces);

    return w_v3_horizontal_add(result);
}

Sphere plane(v3 origin, v3 normal, u32 material_index)
{
    float radius = 100000.0f;
    return (Sphere){
        .center = v3_sub(origin, v3_scale(radius, normal)),
        .radius = radius,
        .material_index = material_index,
    };
}

v3 camera_ray_dir(v3 cam_up, v3 cam_dir, float fov_deg, float dx, float dy)
{
    ASSERT_V3_NORMALIZED(cam_dir, 1.0e-3f);

    // dir = -z
    v3 cam_x = v3_normalized(v3_cross(cam_dir, cam_up));
    v3 cam_y = v3_cross(cam_x, cam_dir);

    float fov_rad = M_PI * fov_deg / 180.0f;
    float z = 1.0f / tan(fov_rad * .5f);
    v3 ray = v3_add(v3_add(v3_scale(dx, cam_x), v3_scale(dy, cam_y)), v3_scale(z, cam_dir));

    return v3_normalized(ray);
}

Image alloc_image(u32 width, u32 height)
{
    Image result;
    result.width = width;
    result.height = height;
    result.bytes_per_pixel = 4;

    result.stride = width * result.bytes_per_pixel;
    result.pixels = malloc(width * height * result.bytes_per_pixel);

    return result;
}

float linear_to_srgb(float val)
{
    return val > 1.0f ? 1.0f : pow(val, 1.0f / 2.2f);
}

int worker_func(void* ptr)
{
    WorkQueue* queue = ptr;

    v3 up = {0.0f, 0.0f, 1.0f};
    v3 cam_orig = {0.0f, -13.0f, 5.0f};
    v3 cam_target = {0.0f, 0.0f, 5.0f};
    v3 cam_dir = v3_normalized(v3_sub(cam_target, cam_orig));
    float cam_fov = 60.0f;

    ASSERT_V3_NORMALIZED(cam_dir, 1.0e-3f);

    u32 samples_per_pixel = 64;
    u32 max_bounce_count = 8;

    u32 rng = rand();

    while (queue->next_work_item < queue->item_count) {
        u32 item_index = FETCH_ADD(&queue->next_work_item, 1);
        printf("%d%%...\r", 100 * queue->next_work_item / queue->item_count);
        fflush(stdout);

        WorkItem* item = &queue->items[item_index];

        float cam_ratio = (float)item->image.width / item->image.height;

        u32 traced_count = 0;
        for (u32 y = item->y_min; y < item->y_max; y++) {
            for (u32 x = item->x_min; x < item->x_max; x++) {
                u8* px = &item->image.pixels[y * item->image.stride + x * item->image.bytes_per_pixel];

                v3 col = {};
                for (u32 i = 0; i < samples_per_pixel; i++) {
                    float sample_x = cam_ratio * ((x + random_float(&rng)) / item->image.width * 2.0f - 1.0f);
                    float sample_y = -((y + random_float(&rng)) / item->image.height * 2.0f - 1.0f);

                    v3 ray_orig = cam_orig;
                    v3 ray_dir = camera_ray_dir(up, cam_dir, cam_fov, sample_x, sample_y);
                    ASSERT_V3_NORMALIZED(ray_dir, 1.0e-3f);

                    u32 bounce_count;
                    v3 dcol = trace_ray(item->world,
                                        &rng,
                                        w_v3_broadcast(ray_orig),
                                        w_v3_broadcast(ray_dir),
                                        max_bounce_count,
                                        &bounce_count);
                    col = v3_add(col, dcol);
                    traced_count += bounce_count;
                }

                *px++ = 255 * linear_to_srgb(col.x / samples_per_pixel);
                *px++ = 255 * linear_to_srgb(col.y / samples_per_pixel);
                *px++ = 255 * linear_to_srgb(col.z / samples_per_pixel);
                *px++ = 0xFF;
            }
        }

        FETCH_ADD(&item->world->traced_ray_count, traced_count);
    }

    return 0;
}

u64 get_nanoseconds()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        return 1;
    }

    srand(time(0));

    Image img = alloc_image(256, 256);

    World world = {};

    Material materials[] = {
        {.diffuse = {}, .emission = v3_scale(.1f, (v3){1.f, 1.0f, 1.0f})},   // sky
        {.diffuse = (v3){.3f, .3f, .3f}},                                    // gray
        {.diffuse = {}, .emission = v3_scale(20.0f, (v3){1.f, 1.0f, 1.0f})}, // light
        {.diffuse = (v3){.9f, .9f, .9f}},                                    // white
        {.diffuse = (v3){.9f, .2f, .2f}},                                    // red
        {.diffuse = (v3){.2f, .9f, .9f}},                                    // blue
    };

    Sphere spheres[] = {
        {
            .center = (v3){.0f, .0f, 1.5f},
            .radius = 1.5f,
            .material_index = 1,
        },
        {
            .center = (v3){0.0f, 0.0f, 8.5f},
            .radius = .8f,
            .material_index = 2,
        },
        plane((v3){.0f, .0f, .0f}, (v3){.0f, .0f, 1.0f}, 3),        // floor
        plane((v3){0.0f, 0.0f, 10.0f}, (v3){0.0f, 0.0f, -1.0f}, 3), // ceiling
        plane((v3){-5.0f, 0.0f, .0f}, (v3){1.0f, 0.0f, .0f}, 4),    // left wall
        plane((v3){5.0f, 0.0f, .0f}, (v3){-1.0f, 0.0f, .0f}, 5),    // right wall
        plane((v3){.0f, 5.0f, .0f}, (v3){.0f, -1.0f, .0f}, 3),      // back wall
    };

    world.materials = materials;
    world.spheres = spheres;
    world.sphere_count = sizeof(spheres) / sizeof(*spheres);

    u32 tile_width = 32;
    u32 tile_height = tile_width;

    u32 x_tile_count = (img.width + tile_width - 1) / tile_width;
    u32 y_tile_count = (img.height + tile_height - 1) / tile_height;

    WorkQueue queue = {};
    queue.item_count = x_tile_count * y_tile_count;
    queue.items = malloc(sizeof(WorkItem) * queue.item_count);

    u32 items_queued = 0;
    for (u32 tile_y = 0; tile_y < y_tile_count; tile_y++) {
        u32 y_min = tile_y * tile_height;
        u32 y_max = y_min + tile_height;
        if (y_max > img.height) {
            y_max = img.height;
        }

        for (u32 tile_x = 0; tile_x < x_tile_count; tile_x++) {
            u32 x_min = tile_x * tile_width;
            u32 x_max = x_min + tile_width;
            if (x_max > img.width) {
                x_max = img.width;
            }

            WorkItem* item = &queue.items[items_queued++];

            item->world = &world;
            item->image = img;
            item->x_min = x_min;
            item->x_max = x_max;
            item->y_min = y_min;
            item->y_max = y_max;
        }
    }
    ASSERT(items_queued == queue.item_count);

    const u32 worker_count = atoi(argv[1]);
    thrd_t workers[worker_count];

    u64 begin = get_nanoseconds();

    for (u32 i = 0; i < worker_count; i++) {
        int err = thrd_create(&workers[i], worker_func, &queue);

        if (err != thrd_success) {
            return 1;
        }
    }

    for (u32 i = 0; i < worker_count; i++) {
        int res;
        int err = thrd_join(workers[i], &res);

        if (err != thrd_success || res != 0) {
            return 1;
        }
    }

    u64 end = get_nanoseconds();

    u32 elapsed_ms = (end - begin) / (1000ull * 1000ull);
    printf("Took %u ms, traced %lu rays, %f us per ray\n",
           elapsed_ms,
           world.traced_ray_count,
           1000.0f * (float)elapsed_ms / world.traced_ray_count);

    int res = stbi_write_png("output.png", img.width, img.height, img.bytes_per_pixel, img.pixels, img.stride);

    if (!res) {
        return 1;
    }

    return 0;
}
