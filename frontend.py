# ==================== imports ==================== #
import math
import copy

from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.app import App

from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget

from kivy.core.window import Window

from solver import solve_sudoku
from scanner import get_board

# ==================== global vars ==================== #
discovered = False
board = [[0]*9]*9
empty_board = [[0] * 9] * 9
solution_board = [[0]*9]*9
final_board = [[0]*9]*9
showed = False

# resize screen
Window.size = (605, 675)


# ==================== main window ==================== #
class MainWindow(Screen):
    """
    The main screen
    Purpose: explain the user about the application
             and navigate him to the selection screen
    """

    def show_popup_about(self):
        """
        open popup with content about the sudoku game
        """
        layout = FloatLayout()
        close_button = Button(text="Close", size_hint=[0.2, 0.05], pos_hint={"x": 0.4, "y": 0.01})
        layout.add_widget(close_button)
        pop = Popup(content=layout, background='images/about.jpg', title="", separator_height=0)
        close_button.bind(on_press=pop.dismiss)
        pop.open()

    def show_popup_introductions(self):
        """
        open popup with content about the sudoku game
        """
        layout = FloatLayout()
        close_button = Button(text="Close", size_hint=[0.2, 0.05], pos_hint={"x": 0.4, "y": 0.01})
        layout.add_widget(close_button)
        pop = Popup(content=layout, background='images/recommends.jpg', title="", separator_height=0)
        close_button.bind(on_press=pop.dismiss)
        pop.open()


# ==================== selection window ==================== #
def prepare_game(image_source):
    """
    with path to the image, the function:
    1. scan and get the board
    2. create board to solve on it
    3. create the board of the final sulotion
    :param image_source: path to the image that selected
    """
    global board
    global solution_board
    global final_board

    board = get_board(image_source)
    solution_board = copy.deepcopy(board)
    final_board = copy.deepcopy(board)
    solve_sudoku(final_board)


class SelectionWindow(Screen, GridLayout):
    def __init__(self, **kwargs):
        super(SelectionWindow, self).__init__(**kwargs)
        Clock.schedule_once(self.update_filechooser_font, 0)

    def selected(self, filename):
        """
        get the image selected
        :param filename: image file
        :return:
        """
        try:
            image_source = filename[0]
            self.box = FloatLayout()

            self.img = (
                Image(source=image_source, size_hint=(None, None), size=(250, 250), pos_hint={'x': .21, 'y': .2}))
            self.box.add_widget(self.img)

            self.close_btn = (
                Button(text="close", size_hint=(None, None), width=200, height=50, pos_hint={'x': 0, 'y': 0}))
            self.box.add_widget(self.close_btn)

            self.ok_btn = Button(id="ok_btn" ,text="select", size_hint=(None, None), width=200, height=50,
                                 pos_hint={'x': .5, 'y': 0})
            self.box.add_widget(self.ok_btn)

            self.main_pop = Popup(title="Image selected", content=self.box, size_hint=(None, None), size=(450, 400),
                                  auto_dismiss=False, title_size=15)

            self.close_btn.bind(on_press=lambda *args: self.main_pop.dismiss())

            def to_game_screen(*args):
                prepare_game(image_source)
                self.main_pop.dismiss()
                self.manager.add_widget(GameWindow())
                App.get_running_app().root.current = "game"

            self.ok_btn.bind(on_press=to_game_screen)

            self.main_pop.open()

        except:
            pass

    def update_filechooser_font(self, *args):
        """
        update the font of the file names in the file chooser
        :param args:
        :return:
        """
        fc = self.ids['filechooser']
        fc.bind(on_entry_added=self.update_file_list_entry)
        fc.bind(on_subentry_to_entry=self.update_file_list_entry)

    def update_file_list_entry(self, file_chooser, file_list_entry, *args):
        """
        update the font in the file chooser and support in hebrew file names
        :param file_chooser:
        :param file_list_entry:
        :param args:
        :return:
        """
        file_list_entry.ids['filename'].font_name = 'DejaVuSans.ttf'
        if any("\u0590" <= c <= "\u05EA" for c in file_list_entry.ids['filename'].text):
            file_list_entry.ids['filename'].halign = 'right'
            file_list_entry.ids['filename'].text = file_list_entry.ids['filename'].text[::-1]


# ==================== game window ==================== #
class GameWindow(Screen, Widget):
    """
    The Game screen
    Purpose: play from the image that we get from the selection screen
             and show the solution or end the game
    """

    def __init__(self, **kwargs):
        super(GameWindow, self).__init__(**kwargs)
        self.grid = GridLayout()
        self.grid.cols = 1
        self.grid.rows = 2
        self.grid.row_default_height = 582
        self.grid.row_force_default = True
        self.grid.col_default_width = 582
        self.grid.col_force_default = True

        self.big_squares = GridLayout()
        self.big_squares.cols = 3
        self.big_squares.rows = 3
        self.big_squares.spacing = 6
        self.big_squares.padding = 2.5
        self.big_squares.row_default_height = 194
        self.big_squares.row_force_default = True
        self.big_squares.col_default_width = 194
        self.big_squares.col_force_default = True

        for i in range(9):
            self.small_squares = GridLayout()
            self.small_squares.cols = 3
            self.small_squares.rows = 3
            self.small_squares.spacing = 1
            self.small_squares.padding = 1
            self.small_squares.row_default_height = 64.6666667
            self.small_squares.row_force_default = True
            self.small_squares.col_default_width = 64.6666667
            self.small_squares.col_force_default = True

            for j in range(9):
                row, col = calc_location(i, j)
                cell = Cell(row, col)
                self.small_squares.add_widget(cell)

            self.big_squares.add_widget(self.small_squares)

        self.grid.add_widget(self.big_squares)
        btns_grid = SolveBtnGrid()
        self.grid.add_widget(btns_grid)
        self.add_widget(self.grid)


def calc_location(i, j):
    """
    convert (i,j) to (row,col) of the board
    :param i: index of square in board
    :param j: index of cell in square
    :return: row and col in the original board
    """
    row = math.floor(i / 3) * 3 + math.floor(j / 3)
    col = (i % 3) * 3 + (j % 3)
    return row, col


# ==================== manager window ==================== #
class WindowManager(ScreenManager):
    """
    Manager the screens
    """
    pass


# ==================== important widgets ==================== #
class Cell(TextInput):
    """
    Cell of one piece of the sudoku grid
    """
    num = ObjectProperty(None)

    def __init__(self, row, col, **kwargs):
        super(Cell, self).__init__(**kwargs)
        self.row = row
        self.col = col
        self.lock = False
        self.lock = board[row][col] > 0
        if self.lock:
            self.num.text = str(board[row][col])
            self.foreground_color = (0.25, 0.25, 0.25, 1)
        self.loc = 9 * self.row + self.col  # index of the cell - not used yet
        self.bind(focus=self.update_when_discovered)

    def update_num(self):
        """
        update the number in the boards like the number of the cell.
        react according to rules of the game.
        """
        global solution_board
        global board
        global showed

        if self.lock:
            self.num.text = str(board[self.row][self.col])
            return
        self.foreground_color = (0, 0, 0, 1)
        if self.num.text == "":
            solution_board[self.row][self.col] = 0
            board[self.row][self.col] = 0
        if not self.num.text.isnumeric() or int(self.num.text) == 0:
            self.num.text = ""
        elif int(self.num.text) > 9:
            self.num.text = str(math.floor(int(self.num.text) / 10))
        elif not is_valid(self.row, self.col, int(self.num.text)):
            self.foreground_color = (1, 0, 0, 1)
            solution_board[self.row][self.col] = int(self.num.text)  # if the user search solution it will not be exist
        else:
            solution_board[self.row][self.col] = int(self.num.text)
            board[self.row][self.col] = int(self.num.text)
            if is_board_complete(board) and not showed:
                showed = True
                show_victory_popup()

    def update_when_discovered(self, instance, text):
        """
        in solve mode on focus in cell, its update the value of the cell according to the solution
        """
        global discovered
        global solution_board
        global board
        global showed

        if discovered:
            self.num.text = str(solution_board[self.row][self.col])
            board[self.row][self.col] = solution_board[self.row][self.col]
            if is_board_complete(board) and not showed:
                showed = True
                show_victory_popup()


class SolveBtnGrid(GridLayout):
    """
    contain buttons of solve mode and end game.
    """
    txt = ObjectProperty(None)

    def enter_solve_mode(self):
        """
        change the state of the game to permit discover the user the answer, and update the solution_board
        """
        global discovered
        global solution_board

        solve_sudoku(solution_board)
        discovered = True
        self.txt.text = "When you exit from solve mode\nyou continue to play as usual"

    def exit_solve_mode(self):
        """
        change the state of the game to prohibit discover the user the answer
        """
        global discovered

        discovered = False
        self.txt.text = "In solve mode any field you\ntouch explore the right number"


# ==================== helper functions ==================== #
def is_valid(row, col, num):
    """
    check if can put num in the board
    :param row: row in board
    :param col: col in board
    :param num: num value
    :return: true if there is no same number in row and col and the square of num
    """
    # check row
    for i in range(9):
        if solution_board[row][i] == num and i != col:
            return False

    # check col
    trans_board = list(map(list, zip(*solution_board)))
    for j in range(9):
        if trans_board[col][j] == num and j != row:
            return False

    # check square
    start_row = math.floor(row / 3) * 3
    start_col = math.floor(col / 3) * 3
    for i in range(3):
        for j in range(3):
            if solution_board[start_row + i][start_col + j] == num and not (
                    start_row + i == row and start_col + j == col):
                return False

    return True


def is_board_complete(current_board):
    """
    check if current_board is complete like the correct solution
    :param current_board: board
    :return: true if the board is complete with the solution
    """
    global final_board
    for row in final_board:
        if 0 in row:
            return False
    return current_board == final_board


def show_victory_popup():
    """
    open popup with victory message
    """
    global showed
    showed = True

    layout = FloatLayout()
    close_button = Button(text="Close", size_hint=[0.2, 0.05], pos_hint={"x": 0.4, "y": 0.01})
    layout.add_widget(close_button)
    pop = Popup(content=layout, background='images/victory.png', title="", separator_height=0, size_hint=[.6, .6])
    close_button.bind(on_press=pop.dismiss)
    pop.open()


# ==================== application ==================== #
class MyGame(App):
    """
    The game application
    """

    def build(self):
        pass