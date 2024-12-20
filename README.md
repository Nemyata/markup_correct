# markup_correct

Проект **markup_correct** предназначен для автоматической обработки изображений и их разметки. Программа обрезает лишние части изображений и преобразует координаты разметки в соответствии с новыми размерами изображения, после чего сохраняет новые изображения и обновленные лейблы в указанную директорию.

## Структура проекта

### Входные данные

Исходная папка должна содержать две поддиректории:
- `image` — папка с исходными изображениями.
- `labels` — папка с файлами разметки, соответствующими изображениям.

### Выходные данные

В выбранной папке для сохранения результатов будут созданы две поддиректории:
- `image` — для сохранения новых изображений после обработки.
- `labels` — для сохранения обновленных файлов разметки.

## Использование

1. Убедитесь, что у вас есть папка с изображениями и соответствующими файлами разметки. Эти папки должны находиться внутри директории, путь к которой указан в переменной `folder_path`.
   
2. Укажите путь к директории, куда будут сохраняться новые изображения и файлы разметки, в переменной `save_directory`.

3. Запустите скрипт, который выполнит следующие действия:
   - Загрузит изображения из папки `image` и файлы разметки из папки `labels` из директории, указанной в `folder_path`.
   - Обрежет изображения и скорректирует разметку под новые координаты.
   - Сохранит обработанные изображения и разметку в папки `image` и `labels` внутри директории, указанной в `save_directory`.

### Пример запуска

```python
folder_path = "/path/to/source/folder"  # Укажите путь к исходной папке
save_directory = "/path/to/save/folder"  # Укажите путь к папке для сохранения результатов

# Запустите скрипт
python markup_correct.py
