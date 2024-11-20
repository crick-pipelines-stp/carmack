#!/usr/bin/env python
"""
Helper functions for updating version
"""

import subprocess

import toml


def get_version():
    """
    Generate a version string based on the latest Git tag and the current branch.

    This function uses `git describe --tags` to get the most recent tag and the number
    of commits since that tag. If the current branch is not `main`, it appends `-dev`
    to the version string. If no tag is found, it defaults to `0.1`.

    Returns:
        str: The generated version string.
    """
    try:
        tag = subprocess.check_output(["git", "describe", "--tags"]).strip().decode("utf-8")
        if "-" in tag:
            tag, commits, _ = tag.split("-")
            version = f"{tag}.{commits}"
        else:
            version = tag
    except subprocess.CalledProcessError:
        version = "0.1"

    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")
    if branch != "main":
        version += "-dev"

    return version


def update_pyproject_version():
    """
    Update the version field in the `pyproject.toml` file based on the current Git state.

    This function generates a version string using `get_version()` and then updates the
    `version` field in the `pyproject.toml` file. The updated version is then saved back
    to the file.
    """
    version = get_version()

    # Load pyproject.toml
    with open("pyproject.toml", "r", encoding="utf-8") as file:
        pyproject_data = toml.load(file)

    # Update the version field
    pyproject_data["project"]["version"] = version

    # Write back the changes to pyproject.toml
    with open("pyproject.toml", "w", encoding="utf-8") as file:
        toml.dump(pyproject_data, file)

    print(f"Updated pyproject.toml version to {version}")


if __name__ == "__main__":
    update_pyproject_version()
