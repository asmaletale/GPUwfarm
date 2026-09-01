"""
Async population logger: writes genomes and fitnesses to HDF5 via a background thread.

The GA loop calls log() and moves on immediately; the writer thread drains the
queue and appends to the HDF5 file without blocking the optimisation.
"""
import queue
import threading
import numpy as np
import h5py
import hdf5plugin


class AsyncPopulationLogger:
    """
    Background-thread HDF5 writer for per-generation population data.

    Datasets (LZ4-compressed, chunked at 1 generation per chunk):
        genomes   — (n_gens, n_individuals, genome_size) float32
        fitnesses — (n_gens, n_individuals)              float32
    """

    def __init__(self, filepath: str, n_individuals: int, genome_size: int) -> None:
        self._filepath = filepath
        self._n_individuals = n_individuals
        self._genome_size = genome_size
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def log(
        self,
        generation: int,
        genomes: np.ndarray,
        fitnesses: np.ndarray,
        objectives: np.ndarray | None = None,
    ) -> None:
        """Queue one generation. Arrays are copied to avoid races with the GA loop."""
        obj_copy = objectives.copy() if objectives is not None else None
        self._queue.put((generation, genomes.copy(), fitnesses.copy(), obj_copy))

    def close(self) -> None:
        """Signal the writer to finish and wait for it to flush."""
        self._queue.put(None)
        self._thread.join()

    def _writer(self) -> None:
        with h5py.File(self._filepath, "w") as f:
            genomes_ds = f.create_dataset(
                "genomes",
                shape=(0, self._n_individuals, self._genome_size),
                maxshape=(None, self._n_individuals, self._genome_size),
                dtype=np.float32,
                chunks=(1, self._n_individuals, self._genome_size),
                **hdf5plugin.LZ4(),
            )
            fitnesses_ds = f.create_dataset(
                "fitnesses",
                shape=(0, self._n_individuals),
                maxshape=(None, self._n_individuals),
                dtype=np.float32,
                chunks=(1, self._n_individuals),
                **hdf5plugin.LZ4(),
            )
            objectives_ds = None
            while True:
                item = self._queue.get()
                if item is None:
                    break
                generation, genomes, fitnesses, objectives = item
                new_size = generation + 1
                if genomes_ds.shape[0] < new_size:
                    genomes_ds.resize(new_size, axis=0)
                    fitnesses_ds.resize(new_size, axis=0)
                genomes_ds[generation] = genomes.astype(np.float32)
                fitnesses_ds[generation] = fitnesses.astype(np.float32)
                if objectives is not None:
                    if objectives_ds is None:
                        n_obj = objectives.shape[1]
                        objectives_ds = f.create_dataset(
                            "objectives",
                            shape=(0, self._n_individuals, n_obj),
                            maxshape=(None, self._n_individuals, n_obj),
                            dtype=np.float32,
                            chunks=(1, self._n_individuals, n_obj),
                            **hdf5plugin.LZ4(),
                        )
                    if objectives_ds.shape[0] < new_size:
                        objectives_ds.resize(new_size, axis=0)
                    objectives_ds[generation] = objectives.astype(np.float32)
