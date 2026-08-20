from app.tools.filesystem.list_files import list_files, tool as list_files_tool
from app.tools.filesystem.read_file import read_file, tool as read_file_tool
from app.tools.filesystem.write_file import write_file, tool as write_file_tool


TOOLS = [
    list_files_tool,
    read_file_tool,
    write_file_tool,
]