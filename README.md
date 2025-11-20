# `tidy3d-notebooks`

You can see our [pretty example library documentation](https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/docs/index.html) to explore each of these notebooks in more detail.

## Development Guidance

- Make sure all your internal API references start with ``tidy3d.<your_reference>``
- In notebooks, always have absolute links, otherwise the links will break when the user downloads them.
- The `develop` branch is syced to the state of the webcenter in order to enable new notebooks to be featured quickly. 
- If you are developing a notebook using a version that is not the latest official version (i.e. 2.9.1 vs the python client `develop` branch) make sure to merge it only after the python client has been released. Or use a `pre/2.10` branch.

These instructions are valid as of Nov 2025, but may change depending on structural planned changes and you can verify with the team.


## Common Documentation References

| API Resource       | URL                                                                                     |
|--------------------|-----------------------------------------------------------------------------------------|
| Installation Guide | [https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html](https://docs.flexcompute.com/projects/tidy3d/en/latest/install.html) |
| Documentation      | [https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html](https://docs.flexcompute.com/projects/tidy3d/en/latest/index.html)         |
| Example Library    | [https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/docs/index.html](https://docs.flexcompute.com/projects/tidy3d/en/latest/notebooks/docs/index.html) |
| FAQ                | [https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/index.html](https://docs.flexcompute.com/projects/tidy3d/en/latest/faq/docs/index.html)             |


## Related Source Repositories

| Name               | Repository                                      |
|--------------------|-------------------------------------------------|
| Source Code        | https://github.com/flexcompute/tidy3d           |
| Example Notebooks  | https://github.com/flexcompute/tidy3d-notebooks |
| FAQ Source Code    | https://github.com/flexcompute/tidy3d-faq       |
