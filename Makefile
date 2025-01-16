# Check if we're running in Jenkins
ifdef JENKINS_URL
	# Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
	# For local dev, fetch from GitHub
	MAKE_INCLUDES := .make-includes
endif

PACKAGE_NAME = vivarium
# Include the makefiles
include $(MAKE_INCLUDES)/base.mk

$(MAKE_INCLUDES)/%.mk:
	mkdir -p $(MAKE_INCLUDES)
	curl -s -o $@ https://raw.githubusercontent.com/ihmeuw/vivarium_build_utils/feature/pnast/mic-5587-shared-makefiles/resources/makefiles/$*.mk