cd /home/j/Project/Cost_Effectiveness/dev/test_tmp
source test_environment/bin/activate
git clone ssh://git@stash.ihme.washington.edu:7999/cste/ceam.git
cd ceam
python setup.py develop
git checkout "${bamboo.planRepository.revision}"
git branch
py.test
e=$?
cd /home/j/Project/Cost_Effectiveness/dev/test_tmp
rm -rf ceam

exit $e
