from nion.ui import Widgets
from nion.ui import UserInterface


class SectionWidget:

    def __init__(self, ui, title):
        self.__ui = ui
        self.__section_content_column = self.__ui._ui.create_column_widget()
        self.__section_widget = Widgets.SectionWidget(self.__ui._ui, title, self.__section_content_column, 'test')
        self.column = ui.create_column_widget()
        self.__section_content_column.add(self.column._widget)

    @property
    def _ui(self):
        return self.__ui

    @property
    def _section_content_column(self):
        return self.__section_content_column

    @property
    def _widget(self):
        return self.__section_widget

    def add_line_edit(self, label, default_text=None, placeholder_text=None):
        row = self._ui.create_row_widget()
        row.add(self._ui.create_label_widget(label))
        row.add_spacing(5)
        line_edit = self._ui.create_line_edit_widget()
        row.add(line_edit)
        line_edit._widget._behavior.placeholder_text = placeholder_text
        line_edit.text = default_text
        self.column.add(row)
        return line_edit

    def add_push_button(self, label, callback):
        row = self._ui.create_row_widget()
        push_button = self._ui.create_push_button_widget(label)
        push_button.on_clicked = callback
        row.add(push_button)
        self.column.add(row)
        return push_button

    def add_combo_box(self, label, items):
        row = self._ui.create_row_widget()
        row.add(self._ui.create_label_widget(label))
        row.add_spacing(5)
        combo_box = self._ui.create_combo_box_widget(items=items)
        row.add(combo_box)
        row.add_stretch()
        self.column.add(row)
        return combo_box

    def add_check_box(self, label):
        row = self._ui.create_row_widget()
        check_box = self._ui.create_check_box_widget(label)
        row.add(check_box)
        self.column.add(row)
        return check_box


class ScrollAreaWidget:

    def __init__(self, ui):
        self.__ui = ui
        self.__scroll_area_widget = ui.create_scroll_area_widget()

    @property
    def _ui(self):
        return self.__ui

    @property
    def _widget(self):
        return self.__scroll_area_widget

    @property
    def content(self):
        return self._widget.content

    @content.setter
    def content(self, value):
        self._widget.content = value


class LineEditWidget:

    def __init__(self, ui, label):
        self.__ui = ui
        self.__row = ui.create_row_widget()
        self.__row.add(ui.create_label_widget((label)))
        self.__row.add_spacing(5)
        self.__line_edit_widget = self.__ui.create_line_edit_widget()
        self.__row.add(self.__line_edit_widget)

    @property
    def editable(self):
        return self.__line_edit_widget.editable

    @property
    def _behavior(self):
        return self.__line_edit_widget._behavior

    @editable.setter
    def editable(self, value):
        self.__line_edit_widget.editable = value

    @property
    def _widget(self):
        return self.__row

    @property
    def placeholder_text(self):
        return self._behavior.placeholder_text

    @placeholder_text.setter
    def placeholder_text(self, text: str):
        self._behavior.placeholder_text = text

    @property
    def text(self):
        return self.__line_edit_widget.text

    @text.setter
    def text(self, value):
        self.__line_edit_widget.text = value

    @property
    def on_editing_finished(self):
        return self.__line_edit_widget.on_editing_finished

    @on_editing_finished.setter
    def on_editing_finished(self, value):
        self.__line_edit_widget.on_editing_finished = value

    def request_refocus(self):
        self.__line_edit_widget.request_refocus()

    def select_all(self):
        self.__line_edit_widget.select_all()
