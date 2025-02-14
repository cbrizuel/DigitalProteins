import cpd_pymoo as pm
import cpd_pymoo_gdt_cyc_plddt as pm2
import cpd_pymoo_tanh as pm3
import cpd_pymoo_zscore as pm4
import cpd_pymoo_ceng as pm5
import cpd_pymoo_gdt_cm as pm6
import numpy as np
from argparse import ArgumentParser
import traceback



esmfold_url = None
esm_url = None



def create_params(weights: str, num_generations: int) -> dict:
    w = np.array(list(map(float, weights.strip().split(","))))
    return {
        "jobname": None,
        "num_generations": num_generations,#225,
        "population_size": 120,
        "weights": w,
        "target_pdb_path": None,
        "crossover_probability": 1.0,
        "mut_ind_prob": 0.16667,
        "mut_gene_prob": 0.1,
        "tournament_size": 2,
        "save_every_generation": 10,
        "esmfold_url": esmfold_url,
        "esm_url": esm_url,
        "prediction_pdb_file_path": None,
        "sleep_time": 0.0,
        "connection_timeout": 30.0
    }



TARGET_PDBS = [
    "1JBL", # short (<= 16)
    "1K83",
    "1QVK",
    "1QVL",
    "1SKI",
    "1SKK",
    "2LWS",
    "2NDL",
    "2NDM",
    "2NS4",
    "2OTQ",
    "2OX2",
    "2VUM",
    "6DNY",
    "6U7Q",
    "6U7R",
    "6WPV",
    "7M25",
    "7M27",
    "7M28",
    "7M29",
    "7M2A",
    "7M2B",
    "7M2C",
    "2ATG", # long (>16)
    "2JUE",
    "2LWT",
    "2MSO",
    "2N07",
    "2NB5",
    "4TTN",
    "5KWZ",
    "5WOV",
    "6PIP",
    "7K7X",
    "7L53",
    "7L55",
    "7RIH",
    "7RIJ"
]




def get_jobname(algorithm: str, params: dict, target_pdb: str, idx: int) -> str:
    p = params["population_size"]
    c = int(100 * params["crossover_probability"])
    m = int(100 * params["mut_ind_prob"])
    t = params["tournament_size"]
    n = len(params["weights"])
    if n == 3:
        gdt, cyc, plddt = (int(100 * x) for x in params["weights"])
        return f"pymoo_{target_pdb}_{idx}_p{p}_c{c}_m{m}_t{t}_gdt{gdt}_cyc{cyc}_plddt{plddt}"
    elif n == 4:
        cm, cyc, eng, plddt = (int(100 * x) for x in params["weights"])
        return f"pymoo_{target_pdb}_{idx}_p{p}_c{c}_m{m}_t{t}_cm{cm}_cyc{cyc}_eng{eng}_plddt{plddt}"
    elif n == 5 and algorithm == "cm_cyc_eng_tanh_plddt":
        cm, cyc, eng, tanh, plddt = (int(100 * x) for x in params["weights"])
        return f"pymoo_{target_pdb}_{idx}_p{p}_c{c}_m{m}_t{t}_cm{cm}_cyc{cyc}_eng{eng}_tanh{tanh}_plddt{plddt}"
    elif n == 6 and algorithm == "cm_cyc_eng_tanh_z_plddt":
        cm, cyc, eng, tanh, z, plddt = (int(100 * x) for x in params["weights"])
        return f"pymoo_{target_pdb}_{idx}_p{p}_c{c}_m{m}_t{t}_cm{cm}_cyc{cyc}_eng{eng}_tanh{tanh}_z{z}_plddt{plddt}"
    elif n == 6 and algorithm == "cm_cyc_eng_tanh_ceng_plddt":
        cm, cyc, eng, tanh, ceng, plddt = (int(100 * x) for x in params["weights"])
        return f"pymoo_{target_pdb}_{idx}_p{p}_c{c}_m{m}_t{t}_cm{cm}_cyc{cyc}_eng{eng}_tanh{tanh}_ceng{ceng}_plddt{plddt}"
    elif n == 7:
        rmsd, gdt, tm, cm, cyc, eng, plddt = (int(100 * x) for x in params["weights"])
        return f"pymoo_{target_pdb}_{idx}_p{p}_c{c}_m{m}_t{t}_rmsd{rmsd}_gdt{gdt}_tm{tm}_cm{cm}_cyc{cyc}_eng{eng}_plddt{plddt}"
    raise RuntimeError



def save_to_log_file(file_path: str, output: str) -> None:
    with open(file_path, "wt") as f:
        f.write(output)



def process_chunk(prediction_pdb_name: str,
                  chunk_pdbs: list, 
                  thread_idx: int, 
                  num_executions_per_target: int,
                  weights: str,
                  algorithm: str,
                  num_generations: int
                  ) -> None:
    algorithms = {
        "cm_cyc_eng_plddt": pm.run_pymoo_algorithm,
        "gdt_cyc_plddt": pm2.run_pymoo_algorithm,
        "cm_cyc_eng_tanh_plddt": pm3.run_pymoo_algorithm,
        "cm_cyc_eng_tanh_z_plddt": pm4.run_pymoo_algorithm,
        "cm_cyc_eng_tanh_ceng_plddt": pm5.run_pymoo_algorithm,
        "rmsd_gdt_tm_cm_cyc_eng_plddt": pm6.run_pymoo_algorithm
    }
    for i in range(num_executions_per_target):
        for target_pdb in chunk_pdbs:
            ga_params = create_params(weights, num_generations)
            ga_params["jobname"] = get_jobname(algorithm, ga_params, target_pdb, i)
            ga_params["target_pdb_path"] = f"cyclic_peptides_dataset/{target_pdb}.pdb"
            ga_params["prediction_pdb_file_path"] = \
                f"{prediction_pdb_name}_{thread_idx}.pdb"
            try:
                algorithms[algorithm](**ga_params)
                # test_fn(ga_params["jobname"])
            except Exception as e:
                save_to_log_file(ga_params["jobname"] + ".log", 
                                 traceback.format_exc())





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("prediction_pdb_name", type=str)
    parser.add_argument("start", type=int)
    parser.add_argument("--esmfold_url", 
                        "-p", 
                        type=str, 
                        default="http://127.0.0.1:8000/esmfold")
    parser.add_argument("--esm_url", 
                        "-d", 
                        type=str, 
                        default="http://127.0.0.1:8000/esm")
    parser.add_argument("--length", "-l", type=int, default=10)
    parser.add_argument("--num_exec_per_target", "-e", type=int, default=3)
    parser.add_argument("--weights", 
                        "-w", 
                        type=str, 
                        default="0.25,0.25,0.25,0.25")
    parser.add_argument("--algorithm", 
                        "-a", 
                        type=str, 
                        default="cm_cyc_eng_plddt")
    parser.add_argument("--num_generations", "-g", type=int, default=150)
    args = parser.parse_args()

    esmfold_url = args.esmfold_url
    esm_url = args.esm_url

    target_pdb_ids = TARGET_PDBS[args.start : args.start + args.length]
    process_chunk(args.prediction_pdb_name,
                  target_pdb_ids, 
                  0,
                  args.num_exec_per_target,
                  args.weights,
                  args.algorithm,
                  args.num_generations)
