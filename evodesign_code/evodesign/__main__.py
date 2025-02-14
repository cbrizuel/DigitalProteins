from argparse import ArgumentParser
import evodesign.Settings as Settings
from .Exceptions import *
from requests.exceptions import ConnectTimeout
import sys
import json
from .Context import Context





parser = ArgumentParser(prog='evodesign',
                        description='A rudimentary suite of evolutionary '
                                    'algorithms for protein design.')
parser.add_argument('settings_path', 
                    help='path to the JSON file describing the configuration '
                         'of the evolutionary algorithm to be executed')
parser.add_argument('target_pdb_path',
                    help='path to the PDB file that contains the target '
                         'protein backbone')
parser.add_argument('workspace_root',
                    help='path to the folder where all the output files will '
                         'be written')
parser.add_argument('-n', '--num_generations', 
                    type=int, 
                    default=0,
                    help='stops algorithm execution after a the specified '
                         'amount of generations are produced, instead of '
                         'the amount specified in the settings file')
parser.add_argument('-f', '--target_fasta_path',
                    type=str,
                    default=None,
                    help='path to the FASTA file containing the target '
                         'amino acid sequence for when the goal is to find '
                         'a different sequence with the same properties as the '
                         'target protein')
parser.add_argument('-r', '--sequence_restrictions',
                    type=str,
                    default=None,
                    help='path to the JSON file describing the allowed amino acids '
                         'for certain positions in the designed sequences')
parser.add_argument('-p', '--save_prediction_pdbs',
                    action='store_true',
                    default=False,
                    help='indicates if the PDB files of the protein structure '
                         'predictions in the workspace')
parser.add_argument('-s', '--seed_rng',
                    type=int,
                    default=None,
                    help='sets a custom seed for the pseudo-random number generator')
parser.add_argument("-g", "--save_graph_png",
                    action="store_true",
                    default=False,
                    help="indicates if the graph showing how the fitness and "
                         "diversity change over each generation should be saved in a"
                         "PNG file or not saved at all")
args = parser.parse_args()

context = Context.create(args.target_pdb_path, 
                         args.target_fasta_path, 
                         args.num_generations,
                         args.sequence_restrictions)
with open(args.settings_path, 'rt', encoding='utf-8') as json_file:
    settings = json.load(json_file)
algorithm = Settings.parse(settings)
algorithm.setup(context, args.workspace_root, args.seed_rng)

while True:
    try:
        algorithm(args.save_prediction_pdbs, args.save_graph_png)
        break
    except (KeyboardInterrupt, HttpBadRequest, HttpUnknownError):
        fasta_option = f'-f {args.target_fasta_path} ' if args.target_fasta_path else ''
        restrictions_option = f'-r {args.sequence_restrictions} ' \
                              if args.sequence_restrictions \
                              else ''
        print(f'\nINTERRUPTED.\n'
              f'Run `python -m evodesign {args.settings_path} {args.target_pdb_path} '
              f'{args.workspace_root} '
              f'{fasta_option}{restrictions_option}` to resume later.')
        sys.exit(130) # SIGINT
    except (HttpInternalServerError, 
            HttpGatewayTimeout,
            HttpForbidden,
            ConnectTimeout):
        pass
