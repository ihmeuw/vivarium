# Check if we're running in Jenkins
ifdef JENKINS_URL
	# Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
	# For local dev, search in parent directory
	MAKE_INCLUDES := ../vivarium_build_utils/resources/makefiles
endif

PACKAGE_NAME = vivarium

.PHONY: install
install: ## Install setuptools, package, and build utilities
	pip install uv
	uv pip install --upgrade pip setuptools 
	uv pip install -e .[DEV]
	@echo "----------------------------------------"
	@if [ ! -d "../vivarium_build_utils" ]; then \
		# Clone the build utils repo if it doesn't exist. \
		git clone https://github.com/ihmeuw/vivarium_build_utils.git ../vivarium_build_utils; \
	else \
		echo "vivarium_build_utils already exists. Skipping clone."; \
	fi

# Include the makefiles
-include $(MAKE_INCLUDES)/base.mk
-include $(MAKE_INCLUDES)/test.mk

