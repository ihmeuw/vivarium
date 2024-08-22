pipeline_name="vivarium"
conda_env_name="${pipeline_name}-${BUILD_NUMBER}"
conda_env_path="/tmp/${conda_env_name}"
// defaults for conda and pip are a local directory /svc-simsci for improved speed.
// In the past, we used /ihme/code/* on the NFS (which is slower)
shared_path="/svc-simsci"


pipeline {
  // This agent runs as svc-simsci on node simsci-slurm-sbuild-p01.
  // It has access to standard IHME filesystems and singularity
  agent { label "svc-simsci" }

  options {
    // Keep 100 old builds.
    buildDiscarder logRotator(numToKeepStr: "100")

    // Wait 60 seconds before starting the build.
    // If another commit enters the build queue in this time, the first build will be discarded.
    quietPeriod(60)

    // Fail immediately if any part of a parallel stage fails
    parallelsAlwaysFailFast()
  }

  parameters {
    booleanParam(
      name: "DEPLOY_OVERRIDE",
      defaultValue: false,
      description: "Whether to deploy despite building a non-default branch. Builds of the default branch are always deployed."
    )
    booleanParam(
      name: "IS_CRON",
      defaultValue: true,
      description: "Indicates a recurring build. Used to skip deployment steps."
    )
    string(
      name: "SLACK_TO",
      defaultValue: "simsci-ci-status",
      description: "The Slack channel to send messages to."
    )
    booleanParam(
      name: "DEBUG",
      defaultValue: false,
      description: "Used as needed for debugging purposes."
    )
  }

  stages {
    stage("Initialization") {
      steps {
        script {
          // Use the name of the branch in the build name
          currentBuild.displayName = "#${BUILD_NUMBER} ${GIT_BRANCH}"
        }
      }
    }

    stage("Python version matrix") {
      // we don't want to go to deployment if any of the matrix branches fail
      failFast true
      matrix {
        // customWorkspace setting must be ran within a node
        agent {
          node {
              label "svc-simsci"
          }
        }
        axes {
          axis {
              // parallelize by python minor version
              name 'PYTHON_VERSION'
              values "3.9", "3.10", "3.11"
          }
        }

        environment {
            // Get the branch being built and strip everything but the text after the last "/"
            BRANCH = sh(script: "echo ${GIT_BRANCH} | rev | cut -d '/' -f1 | rev", returnStdout: true).trim()
            TIMESTAMP = sh(script: 'date', returnStdout: true)
            // Specify the path to the .condarc file via environment variable.
            // This file configures the shared conda package cache.
            CONDARC = "${shared_path}/miniconda3/.condarc"
            CONDA_BIN_PATH = "${shared_path}/miniconda3/bin"
            // Specify conda env by build number so that we don't have collisions if builds from
            // different branches happen concurrently.
            PYTHON_DEPLOY_VERSION = "3.11"
            CONDA_ENV_NAME = "${conda_env_name}"
            CONDA_ENV_PATH = "${conda_env_path}_${PYTHON_VERSION}"
            // Set the Pip cache.
            XDG_CACHE_HOME = "${shared_path}/pip-cache"
            // Jenkins commands run in separate processes, so need to activate the environment every
            // time we run pip, poetry, etc.
            ACTIVATE = "source ${CONDA_BIN_PATH}/activate ${CONDA_ENV_PATH} &> /dev/null"
        }

        stages {
            stage("Debug Info") {
              steps {
                echo "Jenkins pipeline run timestamp: ${TIMESTAMP}"
                // Display parameters used.
                echo """Parameters:
                DEPLOY_OVERRIDE: ${params.DEPLOY_OVERRIDE}"""

                // Display environment variables from Jenkins.
                echo """Environment:
                ACTIVATE:       '${ACTIVATE}'
                BUILD_NUMBER:   '${BUILD_NUMBER}'
                BRANCH:         '${BRANCH}'
                CONDARC:        '${CONDARC}'
                CONDA_BIN_PATH: '${CONDA_BIN_PATH}'
                CONDA_ENV_NAME: '${CONDA_ENV_NAME}'
                CONDA_ENV_PATH: '${CONDA_ENV_PATH}'
                GIT_BRANCH:     '${GIT_BRANCH}'
                JOB_NAME:       '${JOB_NAME}'
                WORKSPACE:      '${WORKSPACE}'
                XDG_CACHE_HOME: '${XDG_CACHE_HOME}'"""
              }
            }

            stage("Build Environment") {
              environment {
                // Command for activating the base environment. Activating the base environment sets
                // the correct path to the conda binary which is used to create a new conda env.
                ACTIVATE_BASE = "source ${CONDA_BIN_PATH}/activate &> /dev/null"
              }
              steps {
                // The env should have been cleaned out after the last build, but delete it again
                // here just to be safe.
                sh "rm -rf ${CONDA_ENV_PATH}"
                sh "${ACTIVATE_BASE} && make build-env PYTHON_VERSION=${PYTHON_VERSION}"
                // open permissions for test users to create file in workspace
                sh "chmod 777 ${WORKSPACE}"
              }
            }

            stage("Install Package") {
              steps {
                sh "${ACTIVATE} && make install"
              }
            }

            // stage("Dependencies") {
            //   steps {
            //     sh "${ACTIVATE} && make dependencies \"ARGS=${GIT_BRANCH}\""
            //   }
            // }

            // Quality Checks
            stage("Format") {
              steps {
                sh "${ACTIVATE} && make format"
              }
            }

            // stage("Lint") {
            //   steps {
            //     sh "${ACTIVATE} && make lint"
            //   }
            // }

            // Tests
            // removable, if passwords can be exported to env. securely without bash indirection
            stage("Run Integration Tests") {
              steps {
                sh "${ACTIVATE} && make integration"
                publishHTML([
                  allowMissing: true,
                  alwaysLinkToLastBuild: false,
                  keepAll: true,
                  reportDir: "output/htmlcov_integration",
                  reportFiles: "index.html",
                  reportName: "Coverage Report - Integration tests",
                  reportTitles: ''
                ])
              }
            }

            // Build
            stage('Build and Deploy') {
              when {
                expression { "${PYTHON_DEPLOY_VERSION}" == "${PYTHON_VERSION}" }
              }
              stages {
                stage("Build Docs") {
                  steps {
                    sh "${ACTIVATE} && make build-doc"
                  }
                }
                stage("Build Package") {
                  steps {
                    sh "${ACTIVATE} && make build-package"
                  }
                }
              } // stages within build and deploy
            } // build and deploy stage
        } // stages bracket within Python matrix
        post {
          always {
            sh "${ACTIVATE} && make clean"
            sh "rm -rf ${CONDA_ENV_PATH}"
            // Delete the workspace directory.
            deleteDir()
          }
          failure {
            script {
              if (env.BRANCH == "main") {
                channelName = "simsci-ci-status"
              } else {
                channelName = "simsci-ci-status-test"
              }
            }
            // TODO: DM the developer instead of the slack channel
            echo "This build failed on ${GIT_BRANCH}. Sending a failure message to Slack."
            slackSend channel: "#${channelName}",
                        message: ":x: JOB FAILURE: $JOB_NAME - $BUILD_ID\n\n${BUILD_URL}console\n\n<!channel>",
                        teamDomain: "ihme",
                        tokenCredentialId: "slack"
          }
          success {
            script {
              if (params.DEBUG) {
                echo 'Debug is enabled. Sending a success message to Slack.'
                slackSend channel: "#${params.SLACK_TO}",
                          message: ":white_check_mark: (debugging) JOB SUCCESS: $JOB_NAME - $BUILD_ID\n\n${BUILD_URL}console",
                          teamDomain: "ihme",
                          tokenCredentialId: "slack"
              } else {
                echo 'Debug is not enabled. No success message will be sent to Slack.'
              }
            }
          }
        } // post bracket
      } // Python matrix bracket
    } // Python matrix stage bracket
  } // stages bracket
} // pipeline bracket