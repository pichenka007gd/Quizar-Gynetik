#define _GNU_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    size_t pop_size;
    size_t genome_len;
    size_t elitism;
    int *layers;
    size_t layers_count;
    double *genomes;
    double *tmp_genomes;
    double *fitnesses;
} GA;

static double randd(void){
    return (double)rand()/(double)RAND_MAX;
}
static double randn(void){
    double u1=randd(), u2=randd();
    if(u1<1e-12) u1=1e-12;
    return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
}
static size_t compute_genome_len(const int *layers, size_t L){
    size_t s=0;
    for(size_t i=0;i+1<L;++i){
        int in=layers[i], out=layers[i+1];
        s += (size_t)in * (size_t)out;
        s += (size_t)out;
    }
    return s;
}
void* ga_create(size_t pop_size, const int *layers, size_t layers_count, size_t elitism){
    if(!layers || layers_count<2 || pop_size==0) return NULL;
    GA *g=(GA*)malloc(sizeof(GA));
    if(!g) return NULL;
    g->pop_size=pop_size;
    g->layers_count=layers_count;
    g->elitism=elitism;
    g->layers=(int*)malloc(sizeof(int)*layers_count);
    if(!g->layers){ free(g); return NULL; }
    for(size_t i=0;i<layers_count;++i) g->layers[i]=layers[i];
    g->genome_len = compute_genome_len(g->layers, g->layers_count);
    g->genomes=(double*)calloc(pop_size * g->genome_len, sizeof(double));
    g->tmp_genomes=(double*)calloc(pop_size * g->genome_len, sizeof(double));
    g->fitnesses=(double*)calloc(pop_size, sizeof(double));
    if(!g->genomes || !g->tmp_genomes || !g->fitnesses){
        free(g->layers); free(g->genomes); free(g->tmp_genomes); free(g->fitnesses); free(g);
        return NULL;
    }
    srand((unsigned)time(NULL) ^ (unsigned)(uintptr_t)g);
    return (void*)g;
}
void ga_destroy(void* ptr){
    GA* g=(GA*)ptr;
    if(!g) return;
    free(g->layers);
    free(g->genomes);
    free(g->tmp_genomes);
    free(g->fitnesses);
    free(g);
}
void ga_randomize(void* ptr, double minv, double maxv){
    GA* g=(GA*)ptr;
    if(!g) return;
    size_t total=g->pop_size * g->genome_len;
    for(size_t i=0;i<total;++i){
        double r=randd();
        g->genomes[i]=minv + r*(maxv-minv);
    }
}
double* ga_get_genomes(void* ptr){
    GA* g=(GA*)ptr;
    if(!g) return NULL;
    return g->genomes;
}
size_t ga_genome_len(void* ptr){
    GA* g=(GA*)ptr;
    if(!g) return 0;
    return g->genome_len;
}
size_t ga_pop_size(void* ptr){
    GA* g=(GA*)ptr;
    if(!g) return 0;
    return g->pop_size;
}
void ga_predict(void* ptr, size_t agent_idx, const double* inputs, size_t input_len, double* outputs, size_t outputs_len){
    GA* g=(GA*)ptr;
    if(!g) return;
    if(agent_idx >= g->pop_size) return;
    if(input_len != (size_t)g->layers[0]) return;
    if(outputs_len != (size_t)g->layers[g->layers_count-1]) return;
    double *gen = g->genomes + agent_idx * g->genome_len;
    size_t offset = 0;
    double *act_prev = (double*)malloc(sizeof(double) * (size_t)g->layers[0]);
    if(!act_prev) return;
    for(size_t i=0;i<(size_t)g->layers[0];++i) act_prev[i]=inputs[i];
    for(size_t li=0; li+1<g->layers_count; ++li){
        int in = g->layers[li];
        int out = g->layers[li+1];
        double *weights = gen + offset;
        offset += (size_t)in * (size_t)out;
        double *biases = gen + offset;
        offset += (size_t)out;
        double *act_next = (double*)malloc(sizeof(double) * (size_t)out);
        if(!act_next){ free(act_prev); return; }
        #pragma omp parallel for if(out>32)
        for(int j=0;j<out;++j){
            double s=0.0;
            size_t base = (size_t)j * (size_t)in;
            for(int i=0;i<in;++i) s += weights[base + (size_t)i] * act_prev[i];
            s += biases[j];
            if(li + 1 == g->layers_count - 1) act_next[j] = s;
            else act_next[j] = tanh(s);
        }
        free(act_prev);
        act_prev = act_next;
    }
    size_t out_n = g->layers[g->layers_count-1];
    for(size_t i=0;i<out_n;++i) outputs[i] = act_prev[i];
    free(act_prev);
}
void ga_set_fitnesses(void* ptr, const double* fitnesses){
    GA* g=(GA*)ptr;
    if(!g || !fitnesses) return;
    for(size_t i=0;i<g->pop_size;++i) g->fitnesses[i]=fitnesses[i];
}
static size_t tournament_select(GA* g, size_t tsize){
    size_t P=g->pop_size;
    size_t best=(size_t)(randd()*P);
    double bf=g->fitnesses[best];
    for(size_t t=1;t<tsize;++t){
        size_t cand=(size_t)(randd()*P);
        if(g->fitnesses[cand] > bf){ best=cand; bf=g->fitnesses[cand]; }
    }
    return best;
}
void ga_evolve(void* ptr, double crossover_rate, double mutation_rate, double mutation_strength){
    GA* g=(GA*)ptr;
    if(!g) return;
    size_t P=g->pop_size;
    size_t L=g->genome_len;
    size_t K=g->elitism;
    if(K> P) K = P;
    size_t *top_idx = NULL;
    double *top_f = NULL;
    if(K>0){
        top_idx = (size_t*)malloc(sizeof(size_t)*K);
        top_f = (double*)malloc(sizeof(double)*K);
        for(size_t i=0;i<K;++i){ top_idx[i]=0; top_f[i]=-INFINITY; }
        for(size_t i=0;i<P;++i){
            double f=g->fitnesses[i];
            for(size_t j=0;j<K;++j){
                if(f>top_f[j]){
                    for(size_t s=(K==0?0:K-1); s>j; --s){
                        top_f[s]=top_f[s-1];
                        top_idx[s]=top_idx[s-1];
                        if(s==0) break;
                    }
                    top_f[j]=f;
                    top_idx[j]=i;
                    break;
                }
            }
        }
    }
    double *newg = g->tmp_genomes;
    size_t filled = 0;
    for(size_t e=0;e<K;++e){
        memcpy(newg + filled * L, g->genomes + top_idx[e] * L, L * sizeof(double));
        filled++;
    }
    while(filled < P){
        size_t a = tournament_select(g, 3);
        size_t b = tournament_select(g, 3);
        if(randd() < crossover_rate){
            size_t cp = 1 + (size_t)(randd() * (L-1));
            memcpy(newg + filled*L, g->genomes + a*L, cp * sizeof(double));
            memcpy(newg + filled*L + cp, g->genomes + b*L + cp, (L - cp) * sizeof(double));
            filled++;
            if(filled >= P) break;
            memcpy(newg + filled*L, g->genomes + b*L, cp * sizeof(double));
            memcpy(newg + filled*L + cp, g->genomes + a*L + cp, (L - cp) * sizeof(double));
            filled++;
        } else {
            memcpy(newg + filled*L, g->genomes + a*L, L * sizeof(double));
            filled++;
        }
    }
    for(size_t i=K;i<P;++i){
        for(size_t j=0;j<L;++j){
            if(randd() < mutation_rate){
                newg[i*L + j] += randn() * mutation_strength;
            }
        }
    }
    memcpy(g->genomes, newg, P * L * sizeof(double));
    if(top_idx) free(top_idx);
    if(top_f) free(top_f);
}
size_t ga_get_best_index(void* ptr){
    GA* g=(GA*)ptr;
    if(!g) return 0;
    size_t best=0;
    double bf=g->fitnesses[0];
    for(size_t i=1;i<g->pop_size;++i) if(g->fitnesses[i]>bf){ bf=g->fitnesses[i]; best=i; }
    return best;
}
void ga_get_best_genome(void* ptr, double* out_buffer){
    GA* g=(GA*)ptr;
    if(!g || !out_buffer) return;
    size_t idx = ga_get_best_index(ptr);
    memcpy(out_buffer, g->genomes + idx * g->genome_len, g->genome_len * sizeof(double));
}

/* batch predict: inputs_flat: n_samples * input_len (row-major)
   outputs_flat: pop_size * n_samples * out_len (row-major, shape (pop, n_samples, out_len))
*/
void ga_predict_all(void* ptr, const double* inputs_flat, size_t n_samples, double* outputs_flat) {
    GA* g = (GA*)ptr;
    if (!g || !inputs_flat || !outputs_flat) return;
    size_t P = g->pop_size;
    int in_n = g->layers[0];
    int out_n = g->layers[g->layers_count - 1];
    size_t genome_len = g->genome_len;

    int max_layer = 0;
    for (size_t i = 0; i < g->layers_count; ++i) if (g->layers[i] > max_layer) max_layer = g->layers[i];

    #pragma omp parallel for schedule(static)
    for (size_t a = 0; a < P; ++a) {
        double *gen = g->genomes + a * genome_len;
        double *act_prev = (double*)malloc(sizeof(double) * (size_t)max_layer);
        double *act_next = (double*)malloc(sizeof(double) * (size_t)max_layer);
        if (!act_prev || !act_next) {
            if (act_prev) free(act_prev);
            if (act_next) free(act_next);
            continue;
        }
        for (size_t s = 0; s < n_samples; ++s) {
            const double *inp = inputs_flat + s * (size_t)in_n;
            for (int i = 0; i < in_n; ++i) act_prev[i] = inp[i];
            size_t offset = 0;
            for (size_t li = 0; li + 1 < g->layers_count; ++li) {
                int in = g->layers[li];
                int out = g->layers[li+1];
                double *weights = gen + offset;
                offset += (size_t)in * (size_t)out;
                double *biases = gen + offset;
                offset += (size_t)out;
                for (int j = 0; j < out; ++j) {
                    double ssum = 0.0;
                    size_t base = (size_t)j * (size_t)in;
                    for (int i = 0; i < in; ++i) ssum += weights[base + (size_t)i] * act_prev[i];
                    ssum += biases[j];
                    if (li + 1 == g->layers_count - 1) act_next[j] = ssum;
                    else act_next[j] = tanh(ssum);
                }
                /* swap pointers */
                double *tmp = act_prev;
                act_prev = act_next;
                act_next = tmp;
            }
            /* write outputs: index ((a * n_samples + s) * out_n + k) */
            size_t base_out = (a * n_samples + s) * (size_t)out_n;
            for (int k = 0; k < out_n; ++k) outputs_flat[base_out + (size_t)k] = act_prev[k];
        }
        free(act_prev);
        free(act_next);
    }
}
