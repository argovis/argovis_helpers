## Build and release

 - Review metadata in `pyproject.toml`
 - Build a release:
   ```
   docker container run -v $(pwd):/src argovis/argovis_helpers:build-220912 python -m build
   ```
 - Push to testPypi: 
   ```
   docker container run -v $(pwd):/src -it argovis/argovis_helpers:build-220912 twine upload -r testpypi dist/*
   ```
 - Test install and try: `python -m pip install -i https://test.pypi.org/simple argovis_helpers`
 - Push to pypi: 
   ```
   docker container run -v $(pwd):/src -it argovis/argovis_helpers:build-220912 twine upload dist/*
   ```
 - `git add` your new build artifacts under `/dist`
 - Push to github and mint a release matching the version number in `pyproject.toml`.

  