/* This file uses jenkins shared library `vivarium_build_utils`,
found at https://github.com/ihmeuw/vivarium_build_utils
Due to Jenkins convention, importable modules must be stored
in the 'vars' folder.

Updating the shared repo will take affect on the next pipeline invocation.
*/ 

// Load the library with get_vbu_version function
@Library("get_vbu_version@main") _
// Load the appropriate version of vivarium_build_utils
library("vivarium_build_utils@${get_vbu_version()}")

reusable_pipeline(
    scheduled_branches: ["main"], 
    upstream_repos: ["layered_config_tree"]
)
