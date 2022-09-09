#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>
#include <threads.h>
#include <time.h>

#include <unistd.h>

#include "base_types.h"
#include "logging.h"
#include "my_assert.h"
#include "simd.h"
#include "stb_image_write.h"
#include "util.h"

#include "platform.h"

#define FETCH_ADD(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)
#define COMPARE_AND_SWAP(ptr, oldval, newval)                                                                          \
    __atomic_compare_exchange_n(ptr, &oldval, newval, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)


typedef struct {
    v3 emission;
    v3 diffuse;
    float polish; // 0 = mirror, 1 = diffuse
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

    v3 cam_orig;
    v3 cam_up;
    v3 cam_target;
    float cam_fov;
} World;

typedef struct FilmPixel {
    v3 color;
    u32 samples;
} FilmPixel;

typedef struct Film {
    u32 width;
    u32 height;

    FilmPixel* pixels;
} Film;

typedef struct {
    w_v3 point;
    w_v3 tangent1; // in the plane of the outgoing ray
    w_v3 tangent2; // (t1, t2, normal) is right-handed
    w_v3 normal;
    w_u32 material_index;
} Intersect;

typedef struct {
    u32 x_min;
    u32 x_max;
    u32 y_min;
    u32 y_max;

    u32 samples;
} WorkItem;

typedef struct {
    u32 img_width;
    u32 img_height;

    u32 tile_width;
    u32 tile_height;

    u32 samples_per_pixel;
    u32 max_bounce_count;

    u32 worker_count;
} Config;

typedef struct {
    WorkItem* items;
    volatile u32 front;
    volatile u32 back;
} WorkQueue;

typedef struct {
    WorkQueue work_queue;

    volatile u32 items_retired;
    u32 total_item_count;

    Film film;
    mtx_t film_mtx;

    World* world;
    Config config;

    volatile b32 stop_requested;
    volatile u64 traced_ray_count;

    u64 begin_timestamp;
    u64 end_timestamp;
} RenderContext;

typedef struct {
    RenderContext* render_context;
    FilmPixel* pixel_buffer;
} WorkerContext;

u32 enqueue_work_item(WorkQueue* queue, WorkItem item)
{
    u32 idx = FETCH_ADD(&queue->front, 1);
    queue->items[idx] = item;

    return idx;
}

b32 get_next_work_item(WorkQueue* queue, u32* idx)
{
    if (queue->back < queue->front) {
        u32 back = queue->back;
        u32 newback = back + 1;
        if (COMPARE_AND_SWAP(&queue->back, back, newback)) {
            *idx = back;
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

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
    w_u32 res = w_u32_broadcast(0);
    w_float tmin = w_float_broadcast(INFINITY);
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

        w_float b_sq = w_float_mul(b, b);
        w_float c4 = w_float_mul(w_float_broadcast(4.0f), c);

        w_u32 disc_pos_mask = w_float_ge(b_sq, c4);

        if (w_u32_mask_any(disc_pos_mask)) {
            w_u32 tpos = w_u32_and(w_float_lt(b, w_float_broadcast(0.0f)), w_float_gt(c, w_float_broadcast(0.0f)));

            if (w_u32_mask_any(tpos)) {
                w_float disc = w_float_sub(b_sq, c4);
                w_float sqrt_disc = w_float_sqrt(disc);
                w_float t = w_float_mul(w_float_broadcast(-.5f), w_float_add(b, sqrt_disc));

                w_u32 tvalid = w_u32_and(tpos, w_float_lt(t, tmin));

                w_u32 assign_mask = w_u32_and(tvalid, disc_pos_mask);

                if (w_u32_mask_any(assign_mask)) {
                    w_v3 hit_point = w_v3_add(orig, w_v3_scale(t, dir));
                    w_v3 normal = w_v3_div(w_v3_sub(hit_point, center), w_float_broadcast(sphere->radius));

                    w_float_conditional_assign(assign_mask, &tmin, t);
                    w_v3_conditional_assign(assign_mask, &intersect->normal, normal);
                    w_v3_conditional_assign(assign_mask, &intersect->point, hit_point);
                    w_u32_conditional_assign(assign_mask, &intersect->material_index, mat_index);

                    res = w_u32_or(res, assign_mask);
                }
            }
        }
    }

    intersect->tangent2 =
        w_v3_normalized(w_v3_cross(dir, intersect->normal)); // TODO(octave) : fallback for orthogonal hits
    intersect->tangent1 = w_v3_cross(intersect->tangent2, intersect->normal);

    ASSERT_W_V3_NORMALIZED(res, intersect->normal, 1.0e-2f);
    ASSERT_W_V3_NORMALIZED(res, intersect->tangent2, 1.0e-2f);
    ASSERT_W_V3_NORMALIZED(res, intersect->tangent1, 1.0e-2f);

    return res;
}

w_u32 w_u32_xorshift(w_u32* state)
{
    w_u32 x = *state;
    x = w_u32_xor(x, w_u32_shl(x, 13));
    x = w_u32_xor(x, w_u32_shr(x, 17));
    x = w_u32_xor(x, w_u32_shl(x, 5));

    *state = x;

    return x;
}

u32 u32_xorshift(u32* state)
{
    u32 x = *state;
    x ^= x << 13;
    x ^= x >> 13;
    x ^= x << 5;

    *state = x;

    return *state;
}

float random_float(u32* state, float amplitude)
{
    return (amplitude * u32_xorshift(state) * 0x1.0p-32);
}

float random_float_bidir(u32* state, float amplitude)
{
    return (amplitude * u32_xorshift(state) * 0x1.0p-31 - amplitude);
}

w_float w_random_float_bidir(w_u32* state, float amplitude)
{
    w_float s = w_float_broadcast(amplitude * 0x1.0p-31);

    w_u32 rnd = w_u32_xorshift(state);
    w_float x = w_s32_cast_float(rnd); // cast to float, signed : range -2^31, 2^31-1
    x = w_float_mul(x, s);             // divide by 2^31, multiply by amplitude

    return x;
}

w_v3 w_v3_lerp(w_float lambda, w_v3 u, w_v3 v)
{
    w_float oml = w_float_sub(w_float_broadcast(1.0f), lambda);
    return w_v3_add(w_v3_scale(oml, u), w_v3_scale(lambda, v));
}

w_float w_float_lerp(w_float lambda, w_float u, w_float v)
{
    w_float oml = w_float_sub(w_float_broadcast(1.0f), lambda);
    return w_float_add(w_float_mul(oml, u), w_float_mul(lambda, v));
}

w_float w_random_float(w_u32* state, float amplitude)
{
    w_float x = w_random_float_bidir(state, .5f * amplitude);
    x = w_float_add(x, w_float_broadcast(.5f * amplitude));

    return x;
}

void random_uniform_circle(w_u32* rng, w_float* x_out, w_float* y_out)
{
#if 0
    w_float x, y;
    do {
        x = 2.0f * w_random_float(rng) - 1.0f;
        y = 2.0f * w_random_float(rng) - 1.0f;
    } while (x * x + y * y >= 1.0f);

    *x_out = x;
    *y_out = y;
#else
    w_float r = w_float_sqrt(w_random_float(rng, 1.0f));
    w_float theta = w_random_float(rng, 2.0f * M_PI);

    w_float c, s;
    w_float_sincos(theta, &s, &c);

    *x_out = w_float_mul(r, c);
    *y_out = w_float_mul(r, s);
#endif
}

w_v3 random_cosine_weighted(w_u32* rng)
{
    w_v3 res;
    random_uniform_circle(rng, &res.x, &res.y);
    w_float norm_2d_squared = w_float_add(w_float_mul(res.x, res.x), w_float_mul(res.y, res.y));
    res.z = w_float_sqrt(w_float_sub(w_float_broadcast(1.0f), norm_2d_squared));

    return res;
}

w_v3 w_v3_mirrored(w_v3 dir, w_v3 normal)
{
    w_float n = w_v3_dot(dir, normal);
    //u = n * N + t * T;
    // r = -n*N + t* T = u - 2 * n * N
    w_v3 res = w_v3_add(dir, w_v3_scale(w_float_mul(w_float_broadcast(-2.0f), n), normal));
    return res;
}

v3 trace_ray(const World* world, w_u32* rng, w_v3 orig, w_v3 dir, u32 max_depth, u32* bounces)
{
    w_v3 result = {};
    w_v3 attenuation = w_v3_broadcast((v3){1.0f, 1.0f, 1.0f});
    w_u32 w_bounces = w_u32_broadcast(0);

    w_u32 ray_alive_mask = w_u32_broadcast(0xFFFFFFFF);

    u32 depth;
    for (depth = 0; depth < max_depth; depth++) {
        Intersect itx = {};

        w_u32 bounced_mask = find_intersect(world, orig, dir, &itx);

        u32 mat_indices[SIMD_LANES];
        w_u32_read(itx.material_index, mat_indices);

        // TODO(octave) : alignment!
        float emissions[3][SIMD_LANES];
        float diffuses[3][SIMD_LANES];
        float polishes[SIMD_LANES];

        for (u32 lane = 0; lane < SIMD_LANES; lane++) {
            u32 mat_index = mat_indices[lane];
            for (u32 i = 0; i < 3; i++) {
                emissions[i][lane] = world->materials[mat_index].emission.co[i];
                diffuses[i][lane] = world->materials[mat_index].diffuse.co[i];
            }
            polishes[lane] = world->materials[mat_index].polish;
        }

        w_v3 emission = w_v3_write(emissions);
        w_v3 diffuse = w_v3_write(diffuses);
        w_float polish = w_float_write(polishes);

        emission = w_v3_masked(ray_alive_mask, emission);
        result = w_v3_add(result, w_v3_hadamard(attenuation, emission));

        ray_alive_mask = w_u32_and(ray_alive_mask, bounced_mask);

        // bail out if the whole lane is dead
        if (!w_u32_mask_any(ray_alive_mask)) {
            break;
        }

        w_u32 bounce_increment = w_u32_and(ray_alive_mask, w_u32_broadcast(1));
        w_bounces = w_u32_add(w_bounces, bounce_increment);

        // no cosine term : embedded in cosine-weighted sampling
        attenuation = w_v3_hadamard(attenuation, diffuse);

        // compute bounce ray
        orig = itx.point;
        w_v3 mirror = w_v3_mirrored(dir, itx.normal);
        w_v3 rnd = w_v3_in_basis(random_cosine_weighted(rng), itx.tangent1, itx.tangent2, itx.normal);
        dir = w_v3_lerp(polish, rnd, mirror);
        dir = w_v3_normalized(dir);
    }

    *bounces = w_u32_horizontal_add(w_bounces);

    return w_v3_horizontal_add(result);
}

Sphere plane(v3 origin, v3 normal, u32 material_index)
{
    float radius = 10000.0f;
    return (Sphere){
        .center = v3_sub(origin, v3_scale(radius, normal)),
        .radius = radius,
        .material_index = material_index,
    };
}

Film alloc_film(u32 width, u32 height)
{
    Film result;
    result.width = width;
    result.height = height;

    result.pixels = calloc(width * height, sizeof(FilmPixel));

    return result;
}

float linear_to_srgb(float val)
{
    return val > 1.0f ? 1.0f : pow(val, 1.0f / 2.2f);
}

int worker_func(void* ptr)
{
    WorkerContext* wctx = ptr;
    FilmPixel* pixel_buffer = wctx->pixel_buffer;

    RenderContext* ctx = wctx->render_context;

    /* v3 cam_orig = ctx->world->cam_orig; */
    /* v3 cam_target = ctx->world->cam_target; */
    w_v3 cam_up = w_v3_broadcast(ctx->world->cam_up);

    w_v3 cam_dir = w_v3_broadcast(v3_normalized(v3_sub(ctx->world->cam_target, ctx->world->cam_orig)));
    w_v3 cam_x = w_v3_normalized(w_v3_cross(cam_dir, cam_up));
    w_v3 cam_y = w_v3_cross(cam_x, cam_dir);

    float cam_fov_rad = M_PI * ctx->world->cam_fov / 180.0f;
    float cam_depth = 1.0f / tan(cam_fov_rad * .5f);
    float cam_ratio = (float)ctx->film.width / ctx->film.height;

    u32 rng_seed[SIMD_LANES];
    for (u32 i = 0; i < SIMD_LANES; i++) {
        rng_seed[i] = rand();
    }
    w_u32 rng = w_u32_write(rng_seed);

    while (!ctx->stop_requested) {
        u32 item_index = 0;
        if (!get_next_work_item(&ctx->work_queue, &item_index)) {
            continue;
        }

        WorkItem item = ctx->work_queue.items[item_index];

        // clear pixel buffer
        for (u32 i = 0; i < ctx->config.tile_width * ctx->config.tile_height; i++) {
            pixel_buffer[i].samples = 0;
            pixel_buffer[i].color = (v3){};
        }

        u32 traced_count = 0;
        for (u32 y = item.y_min; y < item.y_max; y++) {
            for (u32 x = item.x_min; x < item.x_max; x++) {
                FilmPixel* px = &pixel_buffer[(y - item.y_min) * ctx->config.tile_width + (x - item.x_min)];

                for (u32 i = 0; i < item.samples / SIMD_LANES; i++) {
                    float sx = 2.0f * cam_ratio / ctx->film.width;
                    float sy = -2.0f / ctx->film.height;

                    w_float dx = w_random_float(&rng, sx);
                    w_float dy = w_random_float(&rng, sy);

                    w_float sample_x = w_float_add(w_float_broadcast(sx * x - cam_ratio), dx);
                    w_float sample_y = w_float_add(w_float_broadcast(sy * y + 1.0f), dy);

                    w_v3 ray_orig = w_v3_broadcast(ctx->world->cam_orig);

                    w_v3 ray_cam_coords = {
                        sample_x,
                        sample_y,
                        w_float_broadcast(cam_depth),
                    };
                    w_v3 ray_dir = w_v3_in_basis(ray_cam_coords, cam_x, cam_y, cam_dir);
                    ray_dir = w_v3_normalized(ray_dir);

                    u32 bounce_count;
                    v3 dcol =
                        trace_ray(ctx->world, &rng, ray_orig, ray_dir, ctx->config.max_bounce_count, &bounce_count);
                    traced_count += bounce_count;

                    px->color = v3_add(px->color, dcol);
                }

                px->samples += item.samples;
            }
        }

        if (ctx->work_queue.back > item_index) { // TODO : more robust check that we weren't interrupted
            FETCH_ADD(&ctx->traced_ray_count, traced_count);
            FETCH_ADD(&ctx->items_retired, 1);

            // copy to film
            mtx_lock(&ctx->film_mtx);

            for (u32 y = item.y_min; y < item.y_max; y++) {
                for (u32 x = item.x_min; x < item.x_max; x++) {
                    FilmPixel* src = &pixel_buffer[(y - item.y_min) * ctx->config.tile_width + (x - item.x_min)];
                    FilmPixel* dst = &ctx->film.pixels[y * ctx->film.width + x];

                    dst->samples += src->samples;
                    dst->color = v3_add(dst->color, src->color);
                }
            }

            mtx_unlock(&ctx->film_mtx);

            printf("Did work item %u/%u (%u%%) ...\r",
                   ctx->items_retired,
                   ctx->total_item_count,
                   100 * ctx->items_retired / ctx->total_item_count);
            fflush(stdout);
        }
    }

    return 0;
}

u64 get_nanoseconds()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

void w_u32_print(w_u32 v)
{
    u32 vals[SIMD_LANES];
    w_u32_read(v, vals);

    printf("[%u] ", SIMD_LANES);
    for (u32 i = 0; i < SIMD_LANES; i++) {
        printf("%u ", vals[i]);
    }
    printf("\n");
}

void w_float_print(w_float v)
{
    float vals[SIMD_LANES];
    w_float_read(v, vals);

    printf("[%u] ", SIMD_LANES);
    for (u32 i = 0; i < SIMD_LANES; i++) {
        printf("%f ", vals[i]);
    }
    printf("\n");
}

void test_rng()
{
    u32 rng_seed[SIMD_LANES];
    for (u32 i = 0; i < SIMD_LANES; i++) {
        rng_seed[i] = rand();
    }
    w_u32 rng = w_u32_write(rng_seed);

    s32 hist_res = 10;
    u32* histogram = calloc(hist_res, sizeof(u32));

    u32 samples = 10000000;
    for (u32 i = 0; i < samples / SIMD_LANES; i++) {
        w_float x = w_random_float(&rng, 1.0f);
        float xv[4];
        w_float_read(x, xv);

        for (u32 l = 0; l < SIMD_LANES; l++) {
            s32 bucket = floor(xv[l] * hist_res);

            if (bucket < 0) {
                bucket = 0;
            } else if (bucket >= hist_res) {
                bucket = hist_res;
            }

            histogram[bucket]++;
        }
    }

    for (s32 bucket = 0; bucket < hist_res; bucket++) {
        printf("%f ", (float)histogram[bucket] / samples);
    }

    printf("\n");
    free(histogram);
}

#define MAX_WORKER_COUNT 256

static void reset_render(RenderContext* ctx)
{
    // TODO(octave) : Wait for current tasks to finish

    ctx->items_retired = 0;
    ctx->work_queue.back = 0;
    ctx->traced_ray_count = 0;

    ctx->begin_timestamp = get_nanoseconds();
    ctx->end_timestamp = 0;

    // copy to film
    mtx_lock(&ctx->film_mtx);
    memset(ctx->film.pixels, 0, sizeof(FilmPixel) * ctx->film.width * ctx->film.height);
    mtx_unlock(&ctx->film_mtx);
}

int main(int /* argc */, const char* argv[])
{
    srand(time(0));

    u32 simplify = 4;
    Config config = {
        .img_width = 2048 / simplify,
        .img_height = 2048 / simplify,
        .tile_width = 64,
        .tile_height = 64,
        .worker_count = 12,
        .max_bounce_count = 20,
        .samples_per_pixel = 65536,
    };

    u32 samples_per_round = SIMD_LANES;
    u32 samples_round = ((config.samples_per_pixel + samples_per_round - 1) / samples_per_round);
    config.samples_per_pixel = samples_round * samples_per_round;

    platform_init(argv[0]);
    platform_init_window("ray", config.img_width, config.img_height);

    Film film = alloc_film(config.img_width, config.img_height);

    World world = {
        .cam_orig = (v3){.0f, -16.0f, 4.0f},
        .cam_fov = 45.0f,
        .cam_target = (v3){.0f, 0.0f, 5.0f},
        .cam_up = (v3){.0f, .0f, 1.0f},
    };

    Material materials[] = {
        [0] = {.diffuse = {}, .emission = v3_scale(.1f, (v3){1.f, 1.0f, 1.0f})},   // sky
        [1] = {.diffuse = (v3){.3f, .3f, .3f}},                                    // gray
        [2] = {.diffuse = {}, .emission = v3_scale(20.0f, (v3){1.0f, 1.0f, .7f})}, // light
        [3] = {.diffuse = (v3){.9f, .9f, .9f}},                                    // white
        [4] = {.diffuse = (v3){.9f, .2f, .2f}},                                    // red
        [5] = {.diffuse = (v3){.2f, .9f, .9f}},                                    // blue
        [6] = {.diffuse = (v3){.9f, .6f, .1f}},                                    // yellow
        [7] = {.diffuse = (v3){.9f, .9f, .9f}, .polish = .99f},                    // mirror
        [8] = {.diffuse = {}, .emission = v3_scale(200.0f, (v3){.7f, .2f, .6f})},  // light 2
    };


    Sphere spheres[] = {
        {.center = (v3){0.0f, 0.0f, 10.0f}, .radius = 1.0f, .material_index = 2}, // light
        plane((v3){.0f, .0f, .0f}, (v3){.0f, .0f, 1.0f}, 3),                      // floor
        plane((v3){0.0f, 0.0f, 10.0f}, (v3){0.0f, 0.0f, -1.0f}, 3),               // ceiling
        plane((v3){-5.0f, 0.0f, .0f}, (v3){1.0f, 0.0f, .0f}, 4),                  // left wall
        plane((v3){5.0f, 0.0f, .0f}, (v3){-1.0f, 0.0f, .0f}, 5),                  // right wall
        plane((v3){.0f, 10.0f, .0f}, v3_normalized((v3){-.0f, -1.0f, -.0f}), 1),  // back wall
        plane((v3){.0f, -10.0f, .0f}, (v3){.0f, 1.0f, .0f}, 1),                   // front wall
        /* main spheres */
        {.center = (v3){-2.0f, -1.0f, 1.5f}, .radius = 1.5f, .material_index = 7},
        {.center = (v3){2.0f, .0f, 2.5f}, .radius = 2.5f, .material_index = 6},
    };

    world.materials = materials;
    world.spheres = spheres;
    world.sphere_count = sizeof(spheres) / sizeof(*spheres);

    u32 x_tile_count = (film.width + config.tile_width - 1) / config.tile_width;
    u32 y_tile_count = (film.height + config.tile_height - 1) / config.tile_height;

    RenderContext ctx = {};

    ctx.world = &world;
    ctx.film = film;
    mtx_init(&ctx.film_mtx, mtx_plain);

    ctx.config = config;

    ctx.total_item_count = x_tile_count * y_tile_count * samples_round;
    ctx.work_queue.items = malloc(sizeof(WorkItem) * ctx.total_item_count);

    for (u32 i = 0; i < samples_round; i++) {
        for (u32 tile_y = 0; tile_y < y_tile_count; tile_y++) {
            u32 y_min = tile_y * config.tile_height;
            u32 y_max = y_min + config.tile_height;
            if (y_max > film.height) {
                y_max = film.height;
            }

            for (u32 tile_x = 0; tile_x < x_tile_count; tile_x++) {
                u32 x_min = tile_x * config.tile_width;
                u32 x_max = x_min + config.tile_width;
                if (x_max > film.width) {
                    x_max = film.width;
                }

                WorkItem item = {
                    .x_min = x_min,
                    .x_max = x_max,
                    .y_min = y_min,
                    .y_max = y_max,
                    .samples = samples_per_round,
                };

                enqueue_work_item(&ctx.work_queue, item);
            }
        }
    }

    thrd_t workers[MAX_WORKER_COUNT];
    if (config.worker_count > MAX_WORKER_COUNT) {
        config.worker_count = MAX_WORKER_COUNT;
        fprintf(stderr, "Cannot have more than %u worker threads, clamping.\n", MAX_WORKER_COUNT);
    }

    WorkerContext worker_contexts[MAX_WORKER_COUNT] = {};
    for (u32 i = 0; i < config.worker_count; i++) {
        worker_contexts[i].render_context = &ctx;
        worker_contexts[i].pixel_buffer = calloc(config.tile_width * config.tile_height, sizeof(FilmPixel));
    }

    ctx.begin_timestamp = get_nanoseconds();
    ctx.end_timestamp = 0;

    for (u32 i = 0; i < config.worker_count; i++) {
        int err = thrd_create(&workers[i], worker_func, &worker_contexts[i]);

        if (err != thrd_success) {
            return 1;
        }
    }

    PlatformInputInfo input = {};
    Backbuffer bb = platform_get_backbuffer();

    while (!input.exit_requested) {
        platform_handle_input_events(&input);

        if (get_bit(input.keys_pressed, KEY_R)) {
            reset_render(&ctx);
        }

        struct {
            u32 key;
            v3 direction;
        } controls[] = {
            {KEY_W, {0, 1, 0}},
            {KEY_A, {-1, 0, 0}},
            {KEY_S, {0, -1, 0}},
            {KEY_D, {1, 0, 0}},
            {KEY_UP_ARROW, {0, 0, 1}},
            {KEY_DOWN_ARROW, {0, 0, -1}},
        };

        for (u32 i = 0; i < STATIC_ARRAY_COUNT(controls); i++) {
            if (get_bit(input.keys_pressed, controls[i].key)) {
                world.cam_orig = v3_add(world.cam_orig, controls[i].direction);
                reset_render(&ctx);
            }
        }

        for (u32 y = 0; y < film.height; y++) {
            for (u32 x = 0; x < film.width; x++) {
                BackbufferPixel* bbpx = &bb.pixels[y * bb.stride + x];
                FilmPixel* filmpx = &film.pixels[y * film.width + x];

                bbpx->a = 0xff;
                bbpx->r = bbpx->a * linear_to_srgb(filmpx->color.x / filmpx->samples);
                bbpx->g = bbpx->a * linear_to_srgb(filmpx->color.y / filmpx->samples);
                bbpx->b = bbpx->a * linear_to_srgb(filmpx->color.z / filmpx->samples);
            }
        }

        platform_swap_buffers();

        ASSERT(ctx.items_retired <= ctx.total_item_count);

        if (!ctx.end_timestamp && ctx.items_retired == ctx.total_item_count) {
            ctx.end_timestamp = get_nanoseconds();
            u64 elapsed_ns = (ctx.end_timestamp - ctx.begin_timestamp);
            u32 elapsed_ms = elapsed_ns / (1000ull * 1000ull);
            printf("Took %u ms, traced %.3g rays, %.2f worker.ns per ray (%.1f MRay/(worker.s))\n",
                   elapsed_ms,
                   (float)ctx.traced_ray_count,
                   (float)config.worker_count * elapsed_ns / ctx.traced_ray_count,
                   1000.0f * ctx.traced_ray_count / (elapsed_ns * config.worker_count));
        }

        platform_sleep_nanoseconds(10ull * 1000ull * 1000ull);
    }

    ctx.end_timestamp = get_nanoseconds();
    u64 elapsed_ns = (ctx.end_timestamp - ctx.begin_timestamp);
    u32 elapsed_ms = elapsed_ns / (1000ull * 1000ull);
    printf("Took %u ms, traced %.3g rays, %.2f worker.ns per ray (%.1f MRay/(worker.s))\n",
           elapsed_ms,
           (float)ctx.traced_ray_count,
           (float)config.worker_count * elapsed_ns / ctx.traced_ray_count,
           1000.0f * ctx.traced_ray_count / (elapsed_ns * config.worker_count));

    ctx.stop_requested = true;

    for (u32 i = 0; i < config.worker_count; i++) {
        int res;
        int err = thrd_join(workers[i], &res);

        if (err != thrd_success || res != 0) {
            return 1;
        }
    }

    u8* pixels = malloc(film.width * film.height * 4);

    for (u32 y = 0; y < film.height; y++) {
        for (u32 x = 0; x < film.width; x++) {
            u8* px = &pixels[(y * film.width + x) * 4];
            FilmPixel* filmpx = &film.pixels[y * film.width + x];

            px[0] = 0xff * linear_to_srgb(filmpx->color.x / filmpx->samples);
            px[1] = 0xff * linear_to_srgb(filmpx->color.y / filmpx->samples);
            px[2] = 0xff * linear_to_srgb(filmpx->color.z / filmpx->samples);
            px[3] = 0xff;
        }
    }

    time_t t = time(NULL);
    struct tm now = *localtime(&t);

    char filename[512];
    snprintf(filename,
             sizeof(filename),
             "output/output-%04d%02d%02d-%02d%02d%02d.png",
             now.tm_year + 1900,
             now.tm_mon + 1,
             now.tm_mday,
             now.tm_hour,
             now.tm_min,
             now.tm_sec);
    stbi_write_png(filename, film.width, film.height, 4, pixels, film.width * 4);

    int err = execlp("xdg-open", "xdg-open", filename, (char*)0);
    perror(strerror(err));

    return 0;
}
