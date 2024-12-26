/*This file uses jenkins shared repo found at 
https://github.com/ihmeuw/vivarium_build_utils
The first line imports all modules with "_"
The second line calls the standard pipeline from 
the vars folder in the shared repo.
Jenkins needs to be configured globally to use the correct branch
To configure the repo/branch go to
* Manage Jenkins
* Configure System
* Global Pipeline Libraries section
* Library subsection
* Name: The Name for the lib
* Version: The branch you want to use. Note, it will do a look up on the branch, 
  and if it doesn't exist it will throw an error.
* Project Repository: Url to the shared lib
* Credentials: SSH key to access the repo

Updating the shared repo will take affect on the next pipeline invocation.
*/ 
@Library("vivarium_build_utils") _
// reusable_pipeline(pipeline_name: "test_poc_resuable_workflow")
reusable_pipeline(scheduled_branches: "main", test_types: ['integration'])
