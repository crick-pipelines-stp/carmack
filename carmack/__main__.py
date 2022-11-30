#!/usr/bin/env python
""" carmack: Helper tools for analysis of single-cell mutli-omic data """
import logging
import os
import sys

import rich
import rich.console
import rich.logging
import rich.traceback
import rich_click as click

# import nf_core
# import nf_core.bump_version
# import nf_core.create
# import nf_core.download
# import nf_core.launch
# import nf_core.licences
# import nf_core.lint
# import nf_core.list
# import nf_core.modules
# import nf_core.schema
# import nf_core.subworkflows
# import nf_core.sync
# import nf_core.utils

# Set up logging as the root logger
# Submodules should all traverse back to this
log = logging.getLogger()

# # Set up .nfcore directory for storing files between sessions
# nf_core.utils.setup_nfcore_dir()

# # Set up nicer formatting of click cli help messages
# click.rich_click.MAX_WIDTH = 100
# click.rich_click.USE_RICH_MARKUP = True
# click.rich_click.COMMAND_GROUPS = {
#     "nf-core": [
#         {
#             "name": "Commands for users",
#             "commands": ["list", "launch", "download", "licences"],
#         },
#         {
#             "name": "Commands for developers",
#             "commands": ["create", "lint", "modules", "schema", "bump-version", "sync"],
#         },
#     ],
#     "nf-core modules": [
#         {
#             "name": "For pipelines",
#             "commands": ["list", "info", "install", "update", "remove", "patch"],
#         },
#         {
#             "name": "Developing new modules",
#             "commands": ["create", "create-test-yml", "lint", "bump-versions", "mulled", "test"],
#         },
#     ],
#     "nf-core subworkflows": [
#         {
#             "name": "For pipelines",
#             "commands": ["install"],
#         },
#         {
#             "name": "Developing new subworkflows",
#             "commands": ["create", "create-test-yml"],
#         },
#     ],
# }
# click.rich_click.OPTION_GROUPS = {
#     "nf-core modules list local": [{"options": ["--dir", "--json", "--help"]}],
# }

# Set up rich stderr console
stderr = rich.console.Console(stderr=True)
stdout = rich.console.Console()

# Set up the rich traceback
rich.traceback.install(console=stderr, width=200, word_wrap=True, extra_lines=1)


def run_carmack():
    # Print carmack header
    stderr.print("\n\n", highlight=False)
    stderr.print("███████████████████████████████████████████████████████████████████", highlight=False)
    stderr.print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░", highlight=False)
    stderr.print("[white]░░░░░█████╗░░█████╗░██████╗░███╗░░░███╗░█████╗░░█████╗░██╗░░██╗░░░░[white]", highlight=False)
    stderr.print("[white]░░░░██╔══██╗██╔══██╗██╔══██╗████╗░████║██╔══██╗██╔══██╗██║░██╔╝░░░░[white]", highlight=False)
    stderr.print("[white]░░░░██║░░╚═╝███████║██████╔╝██╔████╔██║███████║██║░░╚═╝█████═╝░░░░░[white]", highlight=False)
    stderr.print("[white]░░░░██║░░██╗██╔══██║██╔══██╗██║╚██╔╝██║██╔══██║██║░░██╗██╔═██╗░░░░░[white]", highlight=False)
    stderr.print("[white]░░░░╚█████╔╝██║░░██║██║░░██║██║░╚═╝░██║██║░░██║╚█████╔╝██║░╚██╗░░░░[white]", highlight=False)
    stderr.print("[white]░░░░░╚════╝░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░░░░╚═╝╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░░░░[white]", highlight=False)
    stderr.print("░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░", highlight=False)
    stderr.print("███████████████████████████████████████████████████████████████████", highlight=False)
    stderr.print("\n", highlight=False)
    # stderr.print(
    #     f"[grey39]    carmack version {carmack.__version__} - [link=https://github.com/briscoelab/carmack]https://github.com/briscoelab/carmack[/]",
    #     highlight=False,
    # )
    stderr.print(
        f"[grey25]    carmack version 0.1dev - [link=https://github.com/briscoelab/carmack]https://github.com/briscoelab/carmack[/]",
        highlight=False,
    )
    stderr.print("\n", highlight=False)
    stderr.print("███████████████████████████████████████████████████████████████████", highlight=False)
    stderr.print("\n\n", highlight=False)

    # Launch the click cli
    carmack_cli(auto_envvar_prefix="NFCORE")


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(nf_core.__version__)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Print verbose output to the console.")
@click.option("--hide-progress", is_flag=True, default=False, help="Don't show progress bars.")
@click.option("-l", "--log-file", help="Save a verbose log to a file.", metavar="<filename>")
@click.pass_context
def carmack_cli(ctx, verbose, hide_progress, log_file):
    """
    nf-core/tools provides a set of helper tools for use with nf-core Nextflow pipelines.

    It is designed for both end-users running pipelines and also developers creating new pipelines.
    """
    # Set the base logger to output DEBUG
    log.setLevel(logging.DEBUG)

    # Set up logs to the console
    log.addHandler(
        rich.logging.RichHandler(
            level=logging.DEBUG if verbose else logging.INFO,
            console=rich.console.Console(stderr=True, force_terminal=nf_core.utils.rich_force_colors()),
            show_time=False,
            show_path=verbose,  # True if verbose, false otherwise
            markup=True,
        )
    )

    # Set up logs to a file if we asked for one
    if log_file:
        log_fh = logging.FileHandler(log_file, encoding="utf-8")
        log_fh.setLevel(logging.DEBUG)
        log_fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)-20s [%(levelname)-7s]  %(message)s"))
        log.addHandler(log_fh)

    ctx.obj = {
        "verbose": verbose,
        "hide_progress": hide_progress or verbose,  # Always hide progress bar with verbose logging
    }


# Main script is being run - launch the CLI
if __name__ == "__main__":
    run_carmack()
