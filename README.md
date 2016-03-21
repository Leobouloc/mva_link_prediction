# MVA ALTEGRAD course project

### User instructions
- Place data (both provided and generated) in the `data` folder and reference data with relative paths. To do so, don't hardcode the paths as this can behave differently according to the OS, instead, use:
```
import os
file_name = "example"
file_path = os.path.join('..', 'data', file_name)
```
- If you generate heavy data in a format that is not `.csv`, `.xls` or `.txt`, please add your data type to the .gitignore file before your commit
- Please only upload code that should run without errors.
