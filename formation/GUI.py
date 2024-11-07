import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class RobotGUI:
    final_positions = None  # Class attribute to store final positions

    def __init__(self, master, num_robots=4, init_pos=[[0, 0, 0], [1, 1, 0], [1, -1, 0], [-1, 0, 0]]):
        self.master = master
        self.master.title("Robot Reference Point Selector")

        self.num_robots = num_robots  # Set a fixed number of robots for this example

        self.robot_positions = init_pos  # Example initial positions (x, y, orientation)
        self.reference_points = [None] * self.num_robots
        self.selected_robot = tk.IntVar(value=0)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.plot_initial_positions()

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.master.bind("<Left>", self.rotate_left)
        self.master.bind("<Right>", self.rotate_right)

        self.create_dropdown()
        self.create_done_button()
        self.reference_state_label = None

    def create_dropdown(self):
        frame = tk.Frame(self.master)
        frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        label = tk.Label(frame, text="Select Robot:")
        label.pack(side=tk.LEFT, padx=10, pady=10)

        self.dropdown = ttk.Combobox(frame, textvariable=self.selected_robot, state='readonly')
        self.dropdown['values'] = [f'Robot {i+1}' for i in range(self.num_robots)]
        self.dropdown.set('Robot 1')
        self.dropdown.pack(side=tk.LEFT, padx=10, pady=10)

    def create_done_button(self):
        self.done_button = tk.Button(self.master, text="GO!", command=self.return_final_positions_and_close)
        self.done_button.pack(side=tk.BOTTOM, pady=10)

    def plot_initial_positions(self, color='bo'):
        self.ax.clear()
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-3, 3)
        self.ax.grid(True)
        for i, (x, y, theta) in enumerate(self.robot_positions):
            self.plot_robot(x, y, theta, f'R{i+1}')  # Plot robot position and orientation
            self.ax.text(x, y, f'R{i+1}')
        for i, ref_point in enumerate(self.reference_points):
            if ref_point is not None:
                ref_x, ref_y, ref_theta = ref_point
                self.plot_robot(ref_x, ref_y, ref_theta, f'R{i+1}', color)  # Plot reference point and orientation
        self.ax.set_title("Click to select reference points for robots")
        self.canvas.draw()

    def plot_robot(self, x, y, theta, label, color='bo'):
        robot_size = 0.2
        dx = robot_size * np.cos(theta)
        dy = robot_size * np.sin(theta)
        self.ax.plot([x, x + dx], [y, y + dy], 'k-')  # Plot robot orientation
        self.ax.plot(x, y, color)  # Initial position in blue
        self.ax.text(x, y, label)

    def on_click(self, event):
        if event.inaxes is not None:
            selected_robot_name = self.dropdown.get()
            selected_robot_index = int(selected_robot_name.split()[1]) - 1
            if self.reference_points[selected_robot_index] is not None:
                x, y, theta = self.reference_points[selected_robot_index]
                self.reference_points[selected_robot_index] = [event.xdata, event.ydata, theta]  # Preserve existing orientation
            else:
                self.reference_points[selected_robot_index] = [event.xdata, event.ydata, np.deg2rad(90)]  # Set default orientation if not already set
            self.plot_initial_positions(color='ro')
            self.show_reference_state(selected_robot_index)

    def rotate_left(self, event):
        self.rotate_reference(30)

    def rotate_right(self, event):
        self.rotate_reference(-30)

    def rotate_reference(self, angle):
        selected_robot_name = self.dropdown.get()
        selected_robot_index = int(selected_robot_name.split()[1]) - 1
        if self.reference_points[selected_robot_index] is not None:
            x, y, theta = self.reference_points[selected_robot_index]
            new_theta = (theta + np.deg2rad(angle)) % (2 * np.pi)
            self.reference_points[selected_robot_index] = (x, y, new_theta)
            self.plot_initial_positions(color='ro')
            self.show_reference_state(selected_robot_index)

    def show_reference_state(self, index):
        if self.reference_state_label:
            self.reference_state_label.destroy()
        ref_x, ref_y, ref_theta = self.reference_points[index]
        ref_state_text = f"Reference {index + 1}: (x={ref_x:.2f}, y={ref_y:.2f}), θ={np.rad2deg(ref_theta):.2f}°"
        self.reference_state_label = tk.Label(self.master, text=ref_state_text, anchor="ne")
        self.reference_state_label.pack(side=tk.TOP, padx=10, pady=10)

    def return_final_positions_and_close(self):
        if None in self.reference_points:
            messagebox.showwarning("Warning", "Please select reference points for all robots.")
        else:
            global final_positions
            final_positions = np.array(self.reference_points)
            print("Final positions:", final_positions)
            self.master.quit()  # Quit the Tkinter event loop

def run_gui():
    root = tk.Tk()
    app = RobotGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.master.quit)  # Quit gracefully if window is closed
    # Adjust the width and height of the canvas
    app.canvas.get_tk_widget().config(width=450, height=900)
    root.mainloop()
    return final_positions

if __name__ == "__main__":
    final_positions = run_gui()
    print("Final positions outside the GUI:", final_positions)
