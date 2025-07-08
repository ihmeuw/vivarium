/*This file uses jenkins shared library `vivarium_build_utils`,
found at https://github.com/ihmeuw/vivarium_build_utils
Due to Jenkins convention, importable modules must be stored
in the 'vars' folder.

Updating the shared repo will take affect on the next pipeline invocation.
*/ 

// Determine the vivarium_build_utils version and load the library
library "vivarium_build_utils@${getVBUVersion()}"

// Run the reusable pipeline
reusable_pipeline(
    scheduled_branches: ["main"], 
    upstream_repos: ["layered_config_tree"]
)

//////////////////////
// Helper functions //
//////////////////////

def getVBUVersion(String nodeLabel = 'svc-simsci') {
    // Gets the vivarium_build_utils version using the centralized script.

    def vbuVersion = null
    
    node(nodeLabel) {
        checkout scm
        
        // Download the centralized version resolution script
        sh '''
            curl -sSL https://raw.githubusercontent.com/ihmeuw/vivarium_build_utils/main/resources/scripts/get_vbu_version.py -o get_vbu_version.py
            chmod +x get_vbu_version.py
        '''
        
        // Run the script to get the vivarium_build_utils version
        vbuVersion = sh(
            script: 'python3 get_vbu_version.py',
            returnStdout: true
        ).trim()
        
        echo "Resolved vivarium_build_utils version: ${vbuVersion}"
        
        // Clean up the downloaded script
        sh 'rm -f get_vbu_version.py'
    }
    
    return vbuVersion
}
