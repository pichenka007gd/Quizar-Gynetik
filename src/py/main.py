# src/py/ga_mlp.py
from pathlib import Path
import ctypes
import numpy as np
from typing import List, Sequence, Optional, Tuple

_lib_candidates = [
    Path(__file__).resolve().parent.parent.parent / "main.so",
    Path(__file__).resolve().parent / "main.so",
    Path(__file__).resolve().parent.parent.parent / "libga_mlp.so",
    Path(__file__).resolve().parent / "libga_mlp.so",
]
_lib_path: Optional[Path] = None
for p in _lib_candidates:
    if p.exists():
        _lib_path = p
        break
if _lib_path is None:
    raise FileNotFoundError("shared library not found (tried: {})".format(", ".join(str(p) for p in _lib_candidates)))
_lib = ctypes.CDLL(str(_lib_path))

_lib.ga_create.argtypes = (ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t, ctypes.c_size_t)
_lib.ga_create.restype = ctypes.c_void_p
_lib.ga_destroy.argtypes = (ctypes.c_void_p,)
_lib.ga_randomize.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double)
_lib.ga_get_genomes.argtypes = (ctypes.c_void_p,)
_lib.ga_get_genomes.restype = ctypes.POINTER(ctypes.c_double)
_lib.ga_genome_len.argtypes = (ctypes.c_void_p,)
_lib.ga_genome_len.restype = ctypes.c_size_t
_lib.ga_pop_size.argtypes = (ctypes.c_void_p,)
_lib.ga_pop_size.restype = ctypes.c_size_t
_lib.ga_predict.argtypes = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
)
_lib.ga_predict.restype = None
_lib.ga_set_fitnesses.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double))
_lib.ga_set_fitnesses.restype = None
_lib.ga_evolve.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double)
_lib.ga_evolve.restype = None
_lib.ga_get_best_index.argtypes = (ctypes.c_void_p,)
_lib.ga_get_best_index.restype = ctypes.c_size_t
_lib.ga_get_best_genome.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double))
_lib.ga_get_best_genome.restype = None
_lib.ga_predict_all.argtypes = (
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
)
_lib.ga_predict_all.restype = None

class GeneticMLP:
    def __init__(self, layers: Sequence[int], agents: int = 50, best_agents: int = 1):
        if len(layers) < 2:
            raise ValueError("layers must have at least input and output")
        self.layers = list(layers)
        self.agents = int(agents)
        self.elitism = int(best_agents)
        arr = (ctypes.c_int * len(self.layers))(*self.layers)
        self.obj = _lib.ga_create(ctypes.c_size_t(self.agents), arr, ctypes.c_size_t(len(self.layers)), ctypes.c_size_t(self.elitism))
        if not self.obj:
            raise RuntimeError("Failed to create GA object")
        self.genome_len = int(_lib.ga_genome_len(self.obj))
        self.pop_size = int(_lib.ga_pop_size(self.obj))
        self.fitness: np.ndarray = np.zeros(self.agents, dtype=np.double)

    def randomize(self, minv: float = -1.0, maxv: float = 1.0) -> None:
        _lib.ga_randomize(self.obj, ctypes.c_double(minv), ctypes.c_double(maxv))

    def population(self) -> np.ndarray:
        ptr = _lib.ga_get_genomes(self.obj)
        n = int(self.pop_size * self.genome_len)
        arr = np.ctypeslib.as_array(ptr, shape=(n,))
        return arr.reshape((self.pop_size, self.genome_len))

    def get_agent_weights(self, agent_idx: int) -> np.ndarray:
        pop = self.population()
        return pop[int(agent_idx)].copy()

    def set_agent_weights(self, agent_idx: int, weights: Sequence[float]) -> None:
        w = np.ascontiguousarray(weights, dtype=np.double).ravel()
        if w.size != self.genome_len:
            raise ValueError("weights length mismatch")
        pop = self.population()
        pop[int(agent_idx), :] = w

    def predict(self, agent_idx: int, inputs: Sequence[float]) -> np.ndarray:
        inputs_arr = np.ascontiguousarray(np.array(inputs, dtype=np.double).ravel())
        out_len = self.layers[-1]
        out = np.zeros(out_len, dtype=np.double)
        _lib.ga_predict(self.obj,
                        ctypes.c_size_t(agent_idx),
                        inputs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_size_t(inputs_arr.size),
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        ctypes.c_size_t(out_len))
        return out

    def predict_all(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim == 1:
            inputs = inputs.reshape((1, -1))
        if not inputs.flags['C_CONTIGUOUS'] or inputs.dtype != np.double:
            inputs = np.ascontiguousarray(inputs, dtype=np.double)
        n_samples = inputs.shape[0]
        out_dim = int(self.layers[-1])
        pop = int(self.agents)
        outputs = np.empty((pop, n_samples, out_dim), dtype=np.double, order='C')
        _lib.ga_predict_all(self.obj,
                            inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            ctypes.c_size_t(n_samples),
                            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return outputs

    def set_fitnesses(self, fitnesses: Sequence[float]) -> None:
        arr = np.ascontiguousarray(np.array(fitnesses, dtype=np.double))
        if arr.shape[0] != self.agents:
            raise ValueError("fitnesses length must equal agents")
        self.fitness = arr.copy()
        _lib.ga_set_fitnesses(self.obj, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def evolve(self, crossover: float = 0.8, mutate: float = 0.02, strength: float = 0.1) -> None:
        _lib.ga_evolve(self.obj, ctypes.c_double(crossover), ctypes.c_double(mutate), ctypes.c_double(strength))

    def get_best_index(self) -> int:
        return int(_lib.ga_get_best_index(self.obj))

    def get_best_genome(self) -> np.ndarray:
        L = self.genome_len
        buf = (ctypes.c_double * L)()
        _lib.ga_get_best_genome(self.obj, buf)
        return np.frombuffer(buf, dtype=np.double).copy()

    def save_best(self, path: str) -> None:
        if getattr(self, "fitness", None) is None:
            raise RuntimeError("fitness array is not set")
        best_idx = int(np.argmax(self.fitness))
        pop = self.population()
        best = pop[best_idx].copy()
        np.savez_compressed(path,
                            layers=np.array(self.layers, dtype=np.int32),
                            best_genome=best,
                            best_index=np.int32(best_idx),
                            fitness=np.float64(self.fitness[best_idx]))

    def save_population(self, path: str) -> None:
        pop = self.population().copy()
        np.savez_compressed(path,
                            layers=np.array(self.layers, dtype=np.int32),
                            population=pop,
                            fitness=np.array(self.fitness, dtype=np.double))

    @classmethod
    def load_best(cls, path: str) -> "GeneticMLP":
        data = np.load(path, allow_pickle=True)
        layers = data["layers"].tolist()
        genome = data["best_genome"]
        obj = cls(layers=layers, agents=1)
        obj.randomize(-1.0, 1.0)
        obj.set_agent_weights(0, genome)
        obj.fitness = np.array([float(data.get("fitness", 0.0))], dtype=np.double)
        return obj

    @classmethod
    def load_population(cls, path: str) -> "GeneticMLP":
        data = np.load(path, allow_pickle=True)
        layers = data["layers"].tolist()
        population = data["population"]
        popn = int(population.shape[0])
        obj = cls(layers=layers, agents=popn)
        obj.randomize(-1.0, 1.0)
        # copy all weights into C memory
        dest = obj.population()
        if dest.shape != population.shape:
            raise RuntimeError("population shape mismatch")
        dest[:, :] = population
        obj.fitness = np.array(data.get("fitness", np.zeros(popn)), dtype=np.double)
        return obj

    def destroy(self) -> None:
        if self.obj:
            _lib.ga_destroy(self.obj)
            self.obj = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


class GAController:
    def __init__(self, ga: GeneticMLP):
        self.ga = ga
        self.ga.randomize()

    def get_agent_weights(self, idx: int) -> np.ndarray:
        return self.ga.get_agent_weights(idx)

    def set_agent_weights(self, idx: int, weights: Sequence[float]) -> None:
        self.ga.set_agent_weights(idx, weights)

    def get_population(self) -> np.ndarray:
        return self.ga.population()

    def set_population(self, population: np.ndarray) -> None:
        pop = np.ascontiguousarray(population, dtype=np.double)
        if pop.shape != (self.ga.agents, self.ga.genome_len):
            raise ValueError("population shape mismatch")
        dest = self.ga.population()
        dest[:, :] = pop

    def predict_agent(self, idx: int, inputs: Sequence[float]) -> np.ndarray:
        return self.ga.predict(idx, inputs)

    def predict_agent_batch(self, idx: int, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape((1, -1))
        outputs = self.ga.predict_all(X)  # (agents, samples, out)
        return outputs[int(idx), :, :]

    def evaluate_and_step(self, scores: Sequence[float], crossover: float = 0.8, mutate: float = 0.02, strength: float = 0.1) -> Tuple[int, np.ndarray, float]:
        self.ga.set_fitnesses(scores)
        best_before = int(np.argmax(self.ga.fitness))
        best_f_before = float(self.ga.fitness[best_before])
        self.ga.evolve(crossover=crossover, mutate=mutate, strength=strength)
        best_idx = self.ga.get_best_index()
        best_genome = self.ga.get_best_genome()
        best_score = float(self.ga.fitness[best_idx]) if hasattr(self.ga, "fitness") else 0.0
        return best_idx, best_genome, best_score

    def get_best_index(self) -> int:
        return self.ga.get_best_index()

    def save_best(self, path: str) -> None:
        self.ga.save_best(path)

    def load_population(self, path: str) -> None:
        new = GeneticMLP.load_population(path)
        self.ga.destroy()
        self.ga = new
