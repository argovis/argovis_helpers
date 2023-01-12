## Build and release

 - Review metadata in `pyproject.toml`; while determining if code is release worthy, append an rc tag to the semantic version, like `major.minor.patch.rcxx`
 - Build a release:
   ```
   docker container run -v $(pwd):/src argovis/argovis_helpers:build-220912 python -m build
   ```
 - Push to testPypi: 
   ```
   docker container run -v $(pwd):/src -it argovis/argovis_helpers:build-220912 twine upload -r testpypi dist/<your new rc>
   ```
 - Test install and try: `python -m pip install -i https://test.pypi.org/simple argovis_helpers`
 - If all is well, rebuild without the rc tag and push to pypi: 
   ```
   docker container run -v $(pwd):/src -it argovis/argovis_helpers:build-220912 twine upload dist/<your new release>
   ```
 - `git add` your new build artifacts under `/dist`
 - Push to github and mint a release matching the version number in `pyproject.toml`.

  