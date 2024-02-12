import argparse

from . import equi, gdi, hti, hti_ice, hti_liq, hti_water, ti, ti_water, mti

# from . import gdi


def create_parser():
    parser = argparse.ArgumentParser(
        description="DPTI: An Automatic Workflow Software for Thermodynamic Integration Calculations"
    )
    main_subparsers = parser.add_subparsers(
        title="modules",
        description="the subcommands of dpti",
        help="module-level help",
        dest="module",
        required=True,
    )

    equi.add_module_subparsers(main_subparsers)
    hti.add_module_subparsers(main_subparsers)
    hti_liq.add_module_subparsers(main_subparsers)
    hti_ice.add_module_subparsers(main_subparsers)
    hti_water.add_module_subparsers(main_subparsers)
    ti.add_module_subparsers(main_subparsers)
    ti_water.add_module_subparsers(main_subparsers)
    gdi.add_module_subparsers(main_subparsers)
    mti.add_module_subparsers(main_subparsers)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
