import os
import datetime

current_year = datetime.datetime.now().year
owner = "The Core Team"


def add_license_header(file_path, comment_style):
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            license_snippet = "Licensed under the Apache License, Version 2.0"
            if license_snippet not in content:
                if comment_style == "block":
                    header = (
                        f"/*\n * Copyright (c) {current_year} {owner}\n"
                        " *\n"
                        " * Licensed under the Apache License, Version 2.0;\n"
                        " * You may not use this file except in compliance with the License.\n"
                        " * You may obtain a copy of the License at\n"
                        " *\n"
                        " *     http://www.apache.org/licenses/LICENSE-2.0\n"
                        " *\n"
                        " * Unless required by applicable law or agreed to in writing, software\n"
                        " * distributed under the License is distributed on an 'AS IS' BASIS,\n"
                        " * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
                        " * See the License for the specific language governing permissions and\n"
                        " * limitations under the License.\n */\n\n")
                elif comment_style == "line":
                    header = (f"# Copyright (c) {current_year} {owner}\n"
                              "#\n"
                              f"# Licensed under the Apache License, Version 2.0\n"
                              "# You may not use this file except in compliance with the License.\n"
                              "# You may obtain a copy of the License at\n"
                              "#\n"
                              "#     http://www.apache.org/licenses/LICENSE-2.0\n"
                              "#\n"
                              "# Unless required by applicable law or agreed to in writing, software\n"
                              "# distributed under the License is distributed on an 'AS IS' BASIS,\n"
                              "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n"
                              "# See the License for the specific language governing permissions and\n"
                              "# limitations under the License.\n\n")
                file.seek(0, 0)
                file.write(header + content)
    except FileNotFoundError:
        print(f"File not found: {file_path}")


# 定义文件类型和注释风格的映射
file_map = {
    '.cpp': 'block',
    '.h': 'block',
    '.cu': 'block',
    '.py': 'line',
    '.cmake': 'line'
}

# Traverse all directories and files in the project
for root, dirs, files in os.walk("."):  # Start from the root directory of your project
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext in file_map:
            add_license_header(os.path.join(root, file), file_map[ext])
        elif 'CMakeLists.txt' in file:  # Explicit check for CMakeLists.txt files
            add_license_header(os.path.join(root, file), 'line')
