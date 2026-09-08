from app.tools.filesystem.list_files import list_files, tool as list_files_tool
from app.tools.filesystem.read_file import read_file, tool as read_file_tool
from app.tools.filesystem.write_file import write_file, tool as write_file_tool
from app.tools.filesystem.edit_file import edit_file, tool as edit_file_tool
from app.tools.filesystem.delete_file import delete_file, tool as delete_file_tool
from app.tools.filesystem.find_references import find_references, tool as find_references_tool


TOOLS = [
    list_files_tool,
    read_file_tool,
    write_file_tool,
    edit_file_tool,
    delete_file_tool,
    find_references_tool,
]