# Check if we're running in Jenkins
ifdef JENKINS_URL
	# Files are already in workspace from shared library
	MAKE_INCLUDES := .
else
	# For local dev, search in parent directory
	MAKE_INCLUDES := ../vivarium_build_utils/resources/makefiles
endif

PACKAGE_NAME = vivarium

# Include makefiles from vivarium_build_utils
include $(MAKE_INCLUDES)/base.mk
include $(MAKE_INCLUDES)/test.mk

