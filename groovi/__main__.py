import groovi
import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(prog="groovi (%s)" % groovi.__version__,
                                     usage="groovi <command> --config <file>",
                                     description='Graph learning framework for structural variant discovery')
    parser.add_argument("command", help="groovi command to run (call, train, generate)")
    parser.add_argument('--config', required=True, help='groovi YAML config file')
    parser.add_argument("--version", action="version", version=groovi.__version__)
    args = parser.parse_args()
    if not args.command in ["call", "generate", "train", "train_graph_classification"]:
        print("[ERROR] Unknown command: '%s'" % args.command)
        parser.print_usage()
        exit(1)

    groovi_module = importlib.import_module("groovi.cli." + args.command)
    groovi_module.main(args.config)

if __name__ == "__main__":
    main()