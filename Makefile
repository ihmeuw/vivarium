# Check if we're running in Jenkins
ifdef JENKINS_URL
	# Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
	# For local dev, use the installed vivarium_build_utils package
	MAKE_INCLUDES := $(shell python -c "from vivarium_build_utils.resources import get_makefiles_path; print(get_makefiles_path())")
endif

PACKAGE_NAME = vivarium

# Include makefiles from vivarium_build_utils
include $(MAKE_INCLUDES)/base.mk
include $(MAKE_INCLUDES)/test.mk

