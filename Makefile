# Check if we're running in Jenkins
ifdef JENKINS_URL
# 	Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
# 	For local dev, use the installed vivarium_build_utils package if it exists
# 	First, check if we can import vivarium_build_utils and assign 'yes' or 'no'.
# 	We do this by importing the package in python and redirecting stderr to the null device.
# 	If the import is successful (&&), it will print 'yes', otherwise (||) it will print 'no'.
	VIVARIUM_BUILD_UTILS_AVAILABLE := $(shell python -c "import vivarium_build_utils" 2>/dev/null && echo "yes" || echo "no")
# 	If vivarium_build_utils is available, get the makefiles path or else set it to empty
	ifeq ($(VIVARIUM_BUILD_UTILS_AVAILABLE),yes)
		MAKE_INCLUDES := $(shell python -c "from vivarium_build_utils.resources import get_makefiles_path; print(get_makefiles_path())")
	else
		MAKE_INCLUDES :=
	endif
endif

# Set the package name as the last part of this file's parent directory path
PACKAGE_NAME = $(notdir $(CURDIR))

ifneq ($(MAKE_INCLUDES),) # not empty
# Include makefiles from vivarium_build_utils
include $(MAKE_INCLUDES)/base.mk
include $(MAKE_INCLUDES)/test.mk
else # empty
# Use this help message (since the vivarium_build_utils version is not available)
help:
	@echo
	@echo "For Make's standard help, run 'make --help'."
	@echo
	@echo "Most of our Makefile targets are provided by the vivarium_build_utils"
	@echo "package. To access them, you need to create a development environment first."
	@echo
	@echo "make build-env"
	@echo
	@echo "USAGE:"
	@echo "  make build-env [name=<environment name>] [py=<python version>]"
	@echo
	@echo "ARGUMENTS:"
	@echo "  name [optional]"
	@echo "      Name of the conda environment to create (defaults to <PACKAGE_NAME>)"
	@echo "  py [optional]"
	@echo "      Python version (defaults to latest supported)"
	@echo
	@echo "After creating the environment:"
	@echo "  1. Activate it: 'conda activate <environment_name>'"
	@echo "  2. Run 'make help' again to see all newly available targets"
	@echo
endif

build-env: # Create a new environment with installed packages
#	Validate arguments - exit if unsupported arguments are passed
	@allowed="name py"; \
	for arg in $(filter-out build-env,$(MAKECMDGOALS)) $(MAKEFLAGS); do \
		case $$arg in \
			*=*) \
				arg_name=$${arg%%=*}; \
				if ! echo " $$allowed " | grep -q " $$arg_name "; then \
					allowed_list=$$(echo $$allowed | sed 's/ /, /g'); \
					echo "Error: Invalid argument '$$arg_name'. Allowed arguments are: $$allowed_list" >&2; \
					exit 1; \
				fi \
				;; \
		esac; \
	done
	
#   Handle arguments and set defaults
#	name
	@$(eval name ?= $(PACKAGE_NAME))
#	python version - validate if specified, else get default from json file
	@supported_versions=$$(python -c "import json; print(' '.join(json.load(open('python_versions.json'))))" 2>/dev/null || echo ""); \
	if [ -n "$(py)" ] && ! echo "$$supported_versions" | grep -q "$(py)"; then \
		echo "Error: Python version '$(py)' is not supported. Available: $$(echo $$supported_versions | sed 's/ /, /g')" >&2; \
		exit 1; \
	fi
	@$(eval py ?= $(shell python -c "import json; print(max(json.load(open('python_versions.json')), key=lambda x: tuple(map(int, x.split('.')))))"))
	

	conda create -n $(name) python=$(py) --yes
# 	Bootstrap vivarium_build_utils into the new environment
	conda run -n $(name) pip install vivarium_build_utils
	conda run -n $(name) make install

	@echo
	@echo "Finished building environment"
	@echo "  name: $(name)"
	@echo "  python version: $(py)"
	@echo
	@echo "Don't forget to activate it with:"
	@echo "conda activate $(name)"
	@echo
