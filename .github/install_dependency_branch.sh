#!/bin/bash

# Define variables
dependency_name=$1
branch_name=$2

root_dir=$(pwd)
dependency_branch_name='main'
branch_name_to_check=${branch_name}
iterations=0

while [ "$branch_name_to_check" != "$dependency_branch_name" ] && [ $iterations -lt 20 ]
do
  echo "Checking for ${dependency_name} branch: '${branch_name_to_check}'"
  if
    git ls-remote --exit-code \
    --heads https://github.com/ihmeuw/"${dependency_name}".git "${branch_name_to_check}" == "0"
  then
    dependency_branch_name=${branch_name_to_check}
    echo "Found matching branch: ${dependency_branch_name}"
  else
    echo "Could not find ${dependency_name} branch '${branch_name_to_check}'. Finding parent branch."
    branch_name_to_check="$( \
      git show-branch -a \
      | grep '\*' \
      | grep -v "${branch_name_to_check}" \
      | head -n1 \
      | sed 's/[^\[]*//' \
      | awk 'match($0, /\[[a-zA-Z0-9\/.-]+\]/) { print substr( $0, RSTART+1, RLENGTH-2 )}' \
      | sed 's/^origin\///' \
    )"
    if [ -z "$branch_name_to_check" ]; then
      echo "Could not find parent branch. Will use released version of ${dependency_name}."
      branch_name_to_check="main"
    fi
    echo "Checking out branch: ${branch_name_to_check}"
    git checkout "${branch_name_to_check}"
    iterations=$((iterations+1))
  fi
  done
  git checkout "${branch_name}"

echo "${dependency_name}_branch_name=${dependency_branch_name}" >> "$GITHUB_ENV"

if [ "$dependency_branch_name" != "main" ]; then
  echo "Cloning ${dependency_name} branch: ${dependency_branch_name}"
  cd ..
  git clone --branch="${dependency_branch_name}" https://github.com/ihmeuw/"${dependency_name}".git
  cd "${dependency_name}" || exit
  pip install .
  cd "$root_dir" || exit
fi