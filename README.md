## Build & Release

1. from the root of this project, create and launch your build environment:

```
docker image build -t argovis/helpers:build .
docker container run -it -v $(pwd):/src argovis/helpers:build bash
```

2. get api token at https://test.pypi.org/manage/account/#api-tokens

3. bump version number in `setup.cfg`

4. build and push to pypi, with username == `__token__`, pass == api token fetched above:

```
python3 -m build
python3 -m twine upload --verbose --repository testpypi dist/*<your version number>*
```

  