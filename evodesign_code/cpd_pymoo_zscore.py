from evodesign.Metrics.ContactMapRms import ContactMapRms
from evodesign.Metrics.CyclizationNormalized import CyclizationNormalized
from evodesign.Metrics.Normalization.Reciprocal import Reciprocal
from evodesign.Metrics.Normalization.Sigmoid import Sigmoid
from evodesign.Metrics.Normalization.ModifiedTanh import ModifiedTanh
from evodesign.Metrics.RosettaEnergyFunction import RosettaEnergyFunction
from evodesign.Metrics.RosettaEnergyNormalized import RosettaEnergyNormalized
from evodesign.Metrics.ZScore import ZScore
from evodesign.Metrics.ESM2DescriptorsRemoteApi import ESM2DescriptorsRemoteApi
from evodesign.Prediction.ESMFoldRemoteApi import ESMFoldRemoteApi
import evodesign.Chain as Chain
from evodesign.Context import Context
from evodesign.Exceptions import HttpGatewayTimeout
from requests.exceptions import ConnectTimeout
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.core.mutation import Mutation
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.core.callback import Callback
from pymoo.core.survival import Survival
from pymoo.core.variable import Real
import numpy as np
import numpy.typing as npt
from copy import deepcopy
from datetime import datetime
from typing import Optional
import dill
from Bio.PDB import PPBuilder





def save_rng_state(state: tuple, file_path: str) -> None:
    with open(file_path, "wt", encoding="utf-8") as txt_file:
        txt_file.write(f"{state[0]}\n")
        values = ",".join([ str(v) for v in state[1] ])
        txt_file.write(f"{values}\n")
        for i in range(2, len(state)):
            txt_file.write(f"{state[i]}\n")



def load_rng_state(file_path: str) -> tuple:
    result = []
    i = 0
    for line in open(file_path, "rt", encoding="utf-8"):
        if i == 0:
            result.append(line.strip())
        elif i == 1:
            values = [ int(float(x)) for x in line.strip().split(",") ]
            np_values = np.zeros(len(values), dtype=np.uint32)
            for j in range(len(values)):
                np_values[j] = values[j]
            result.append(np_values)
        elif i == 4:
            result.append(float(line.strip()))
        else:
            result.append(int(float(line.strip())))
        i += 1
    return tuple(result)



def save_algorithm(algorithm, file_path: str) -> None:
    with open(file_path, "wb") as bin_file:
        dill.dump(algorithm, bin_file)



def load_algorithm(file_path: str):
    with open(file_path, "rb") as bin_file:
        algorithm = dill.load(bin_file)
    return algorithm



class CPD(Problem):

    AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))



    def __init__(self,
                 weights: npt.NDArray[np.float64],
                 target_pdb_file_path: str,
                 esmfold_url: str = "http://127.0.0.1/esmfold:8000",
                 esm_url: str = "http://127.0.0.1/esm:8000",
                 prediction_pdb_file_path: Optional[str] = "temp.pdb",
                 sleep_time: float = 0.0,
                 connection_timeout: float = 30.0
                 ) -> None:
        self.ref_structure = Chain.load_structure(target_pdb_file_path)
        for chain_id in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            try:
                self.ref_backbone = Chain.backbone_atoms(self.ref_structure, 
                                                         chain_id=chain_id)
            except KeyError:
                if chain_id == "Z": raise KeyError
                continue
            break
        # TODO leer la secuencia de otra parte porque hay AA no canónicos
        # ppb = PPBuilder()
        # polypeptides = ppb.build_peptides(self.ref_structure)
        # self.ref_sequence = polypeptides[0].get_sequence()
        self.sequence_length = len(self.ref_backbone) // 4
        super().__init__(n_var=self.sequence_length, 
                         n_obj=1, 
                         n_eq_constr=0, 
                         n_ieq_constr=0, 
                         xl=0, 
                         xu=19, 
                         vtype=np.int64)
        self.term_values = None
        self.weights = weights
        self.prediction_pdb_file_path = prediction_pdb_file_path
        self.esmfold = ESMFoldRemoteApi(url=esmfold_url, 
                                        sleep_time=sleep_time,
                                        connection_timeout=connection_timeout)
        # self.esm = ESM2DescriptorsRemoteApi(url=esm_url, 
        #                                     sleep_time=sleep_time,
        #                                     connection_timeout=connection_timeout)
        ref15 = RosettaEnergyFunction()
        self.cm_calc = Reciprocal(metric=ContactMapRms())
        self.cyc_calc = CyclizationNormalized()
        self.eng_calc = Sigmoid(scaling_factor=1.0 / 277.0,
                                offset=-2.0,
                                metric=ref15)
        self.eng_calc2 = ModifiedTanh(scaling_factor=-1.0 / 600.0,
                                      offset=0.865,
                                      metric=ref15)
        self.eng_calc3 = RosettaEnergyNormalized(metric=ZScore(
            mean=849.47, 
            standard_deviation=118.8751, 
            metric=ref15))
        # self.ref_desc = self.esm.compute_descriptor_vectors(self.ref_sequence)



    def _evaluate(self, x, out, *args, **kwargs):
        self.term_values = np.apply_along_axis(self.compute_fitness, 1, x)
        out["F"] = -1.0 * np.average(self.term_values, axis=1, weights=self.weights)
    


    def to_amino_acid_sequence(self, sequence: npt.NDArray[np.int64]) -> str:
        return "".join(self.AMINO_ACIDS[sequence].tolist())
    


    def compute_fitness(self, sequence: npt.NDArray[np.int64]) -> float:
        aa_seq = self.to_amino_acid_sequence(sequence)
        while True:
            try:
                self.esmfold.predict_pdb_file(aa_seq, 
                                              self.prediction_pdb_file_path)
            except (HttpGatewayTimeout, ConnectTimeout):
                continue
            break
        # while True:
        #     try:
        #         model_desc = self.esm.compute_descriptor_vectors(aa_seq)
        #     except (HttpGatewayTimeout, ConnectTimeout):
        #         continue
        #     break
        model_structure = Chain.load_structure(self.prediction_pdb_file_path)
        model_bb = Chain.backbone_atoms(model_structure)
        bfactors = np.array([
            atom.get_bfactor()
            for atom in model_structure.get_atoms()
        ])
        plddt = bfactors.mean()
        if plddt > 1.0: plddt = plddt / 100.0
        # esm_rms = np.sqrt(np.mean((self.ref_desc - model_desc) ** 2, axis=1)).mean()
        eng_temp = self.eng_calc._metric._ref2015(self.prediction_pdb_file_path)
        eng = self.eng_calc._normalize(eng_temp)
        eng2 = self.eng_calc2._normalize(eng_temp)
        eng3 = self.eng_calc3._normalize(self.eng_calc3._metric._z_score(eng_temp))
        cm_rms = self.cm_calc._metric._contact_map_rms(model_bb, 
                                                       self.ref_backbone)
        cm_rms = self.cm_calc._normalize(cm_rms)
        cyc = self.cyc_calc._metric._metric._cyclization(model_bb)
        cyc = self.cyc_calc._metric._z_score(cyc)
        cyc = self.cyc_calc._normalize(cyc)
        return np.array([ cm_rms, cyc, eng, eng2, eng3, plddt ])



class RandomResetting(Mutation):

    ALPHABET = np.array(list(range(20)), dtype=np.int64)



    def __init__(self, prob: float, prob_var: float) -> None:
        super().__init__()
        self.prob = Real(prob, bounds=(0.0, 1.0), strict=(0.0, 1.0))
        self.prob_var = Real(prob_var, bounds=(0.0, 1.0), strict=(0.0, 1.0))



    def _do(self, problem, X, **kwargs):
        mask = np.random.random(X.shape) < self.prob_var.value
        mutations = np.random.choice(self.ALPHABET[1:], size=X.shape)
        Xp = deepcopy(X)
        Xp[mask] = (X[mask] + mutations[mask]) % len(self.ALPHABET)
        return Xp



class SaveProgression(Callback):

    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 sequence_length: int,
                 jobname: Optional[str] = None,
                 save_every_generation: int = 10
                 ) -> None:
        super().__init__()
        shape_3d = (num_generations, population_size, sequence_length)
        shape_2d = (num_generations, population_size)
        self.generations = np.full(shape_3d, -1, np.int64)
        self.fitness_values = np.zeros(shape_2d, np.float64)
        self.term_values = np.full((num_generations, population_size, 6), 
                                   -1,
                                   np.float64)
        self.save_every_generation = save_every_generation
        self.jobname = jobname
        if jobname is None:
            today = datetime.today().strftime("%Y%m%d")
            self.jobname = f"{today}_pymoo_ga"
        self.output_file_path = f"{jobname}.npz"
        self.initial_rng_state_file_path = f"{jobname}_initial_rng.txt"
        self.checkpoint_rng_state_file_path = f"{jobname}_last_rng.txt"
        self.algorithm_file_path = f"{jobname}.bin"



    def notify(self, algorithm):
        generation_id = algorithm.n_iter - 1
        for i, solution in enumerate(algorithm.pop.get("X")):
            for j, x in enumerate(solution):
                self.generations[generation_id][i][j] = x
        for i, y in enumerate(algorithm.pop.get("F")):
            self.fitness_values[generation_id][i] = -1.0 * y[0]
        for i, w in enumerate(algorithm.problem.term_values):
            self.term_values[generation_id][i] = w
        if generation_id % self.save_every_generation == 0:
            # save the data for offline analysis
            # save RNG for reproduction/resuming
            self.save()
            # save the algorithm for resuming later
            temp = algorithm.problem.eng_calc._metric
            algorithm.problem.eng_calc._metric = None
            algorithm.problem.eng_calc2._metric = None
            algorithm.problem.eng_calc3._metric._metric = None
            save_algorithm(algorithm, self.algorithm_file_path)
            algorithm.problem.eng_calc._metric = temp
            algorithm.problem.eng_calc2._metric = temp
            algorithm.problem.eng_calc3._metric._metric = temp



    def save(self):
        np.savez_compressed(self.output_file_path,
                            generations=self.generations,
                            fitness_values=self.fitness_values,
                            term_values=self.term_values)
        save_rng_state(np.random.get_state(), 
                        self.checkpoint_rng_state_file_path)
    


    def load(self):
        data = np.load(self.output_file_path)
        self.generations = data["generations"]
        self.fitness_values = data["fitness_values"]
        self.term_values = data["term_values"]



class GenerationalReplacement(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)



    def _do(self, problem, pop, n_survive=None, **kwargs):
        return pop[n_survive:]



def tournament(pop, P, **kwargs):
    selection_size, tournament_size = P.shape
    selected_parents = np.full(selection_size, -1, dtype=np.int64)
    for i in range(selection_size):
        fitness = np.array([ pop[j].F[0] for j in P[i] ])
        selected_parents[i] = P[i][fitness.argmin()]
    return selected_parents



def run_pymoo_algorithm(jobname: str,
                        num_generations: int, 
                        population_size: int,
                        weights: npt.NDArray[np.float64],
                        target_pdb_path: str,
                        crossover_probability: float,
                        mut_ind_prob: float,
                        mut_gene_prob: float,
                        tournament_size: int = 2,
                        save_every_generation: int = 10,
                        esmfold_url: str = "http://127.0.0.1/esmfold:8000",
                        esm_url: str = "http://127.0.0.1/esm:8000",
                        prediction_pdb_file_path: Optional[str] = "temp.pdb",
                        sleep_time: float = 0.0,
                        connection_timeout: float = 30.0
                        ):
    problem = CPD(weights, 
                  target_pdb_path,
                  esmfold_url,
                  esm_url,
                  prediction_pdb_file_path,
                  sleep_time,
                  connection_timeout)
    sampling = IntegerRandomSampling()
    selection = TournamentSelection(pressure=tournament_size, 
                                    func_comp=tournament)
    crossover = UniformCrossover(prob=crossover_probability)
    mutation = RandomResetting(prob=mut_ind_prob, prob_var=mut_gene_prob)
    survival = GenerationalReplacement()
    termination = MaximumGenerationTermination(num_generations)
    callback = SaveProgression(num_generations, 
                               population_size, 
                               problem.sequence_length,
                               jobname,
                               save_every_generation)
    try:
        # resuming from a previous execution
        algorithm = load_algorithm(callback.algorithm_file_path)
        callback.load()
        state = load_rng_state(callback.checkpoint_rng_state_file_path)
        np.random.set_state(state)
    except FileNotFoundError:
        try:
            # starting fresh but with a previous RNG seed
            state = load_rng_state(callback.initial_rng_state_file_path)
            np.random.set_state(state)
        except FileNotFoundError:
            # starting with a fresh RNG seed
            save_rng_state(np.random.get_state(), 
                           callback.initial_rng_state_file_path)
        algorithm = GA(pop_size=population_size, 
                       n_offsprings=population_size, 
                       sampling=sampling,
                       selection=selection, 
                       crossover=crossover, 
                       mutation=mutation,
                       survival=survival,
                       eliminate_duplicates=False)
    minimize(problem, 
             algorithm, 
             callback=callback, 
             termination=termination,
             verbose=False, 
             copy_algorithm=False)
    callback.save()
