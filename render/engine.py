# ------------------ IMPORTS ------------------

import pygame
import neat
import time
import math
import numpy as np
import pickle
import sys
import os
from typing import Tuple, List
from ai.car_ai import CarAI
from render.car import Car
from render.colors import Color
from render.track import Track
from levels.level_data import LEVELS

class StopTraining(Exception): pass

# ------------------ CLASSES ------------------

class Engine:
    VIEW_W = 1600
    VIEW_H = 900
    FPS = 60
    DEFAULT_FONT = "comicsansms"

    def __init__(self, neat_config_path: str, debug: bool, max_simulations: int):
        self.neat_config_path = neat_config_path
        self.debug = debug
        self.max_simulations = max_simulations
        self.title = "NeuroDrive: NEAT AI Racing Simulator"
        
        pygame.init()
        pygame.display.set_caption(self.title)
        
        # Responsive Window
        self.win_width = 1280
        self.win_height = 720
        self.screen = pygame.display.set_mode((self.win_width, self.win_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        
        # Fixed internal viewport for physics stability
        self.viewport = pygame.Surface((self.VIEW_W, self.VIEW_H))
        
        self.track = Track(self.VIEW_W, self.VIEW_H)
        self.car = Car([0, 0], self.track)
        self.decided_car_pos = None
        self.end_point_pos = None
        
        self.state = "main_menu"
        self.stop_early = False
        self.best_genome = None
        self.neat_config = None
        
        self.hyper_speed = False
        self.global_best_fitness = 0
        self.brush_ui_timer = 0
        self.show_data_viewer = False
        self.notification_text = ""
        self.notification_timer = 0
        self.abort_to_menu = False
        self.ignore_drawing = False
        self.current_level_id = 0
        os.makedirs("csv_data", exist_ok=True)
        os.makedirs("saved_brains", exist_ok=True)

    def draw_main_menu(self):
        self.screen.fill(Color.BG_LIGHT)
        font_title = pygame.font.SysFont(self.DEFAULT_FONT, 72, bold=True)
        font_sub = pygame.font.SysFont(self.DEFAULT_FONT, 20, bold=True)
        
        title = font_title.render("NEURODRIVE", True, Color.TEXT_DARK)
        sub = font_sub.render("Neural Network Racing Simulation", True, Color.ACCENT_BLUE)
        
        self.screen.blit(title, (self.win_width//2 - title.get_width()//2, self.win_height//4 - 50))
        self.screen.blit(sub, (self.win_width//2 - sub.get_width()//2, self.win_height//4 + 10))
        
        # Grid settings
        cols = 4
        rows = 4
        btn_w = 320
        btn_h = 50
        gap_x = 20
        gap_y = 15
        total_w = cols * btn_w + (cols - 1) * gap_x
        total_h = rows * btn_h + (rows - 1) * gap_y
        
        start_x = self.win_width // 2 - total_w // 2
        start_y = self.win_height // 2 - total_h // 2 + 50
        
        mx, my = pygame.mouse.get_pos()
        
        for i in range(16):
            row = i // cols
            col = i % cols
            x = start_x + col * (btn_w + gap_x)
            y = start_y + row * (btn_h + gap_y)
            rect = pygame.Rect(x, y, btn_w, btn_h)
            
            bg_color = Color.ACCENT_BLUE if rect.collidepoint(mx, my) else Color.PANEL_BG
            text_color = Color.WHITE if rect.collidepoint(mx, my) else Color.TEXT_DARK
            
            pygame.draw.rect(self.screen, bg_color, rect, border_radius=8)
            pygame.draw.rect(self.screen, Color.ACCENT_BLUE, rect, 2, border_radius=8)
            
            if i == 0:
                txt = font_sub.render("Draw Custom Track", True, text_color)
            else:
                txt = font_sub.render(f"[{i}] {LEVELS[i]['name']}", True, text_color)
                
            self.screen.blit(txt, (x + btn_w//2 - txt.get_width()//2, y + btn_h//2 - txt.get_height()//2))
            
        inst = font_sub.render("Click any level box to start the simulation!", True, Color.ACCENT_MINT)
        self.screen.blit(inst, (self.win_width//2 - inst.get_width()//2, self.win_height - 100))
        
        clear_t = font_sub.render("Press [C] to Wipe Saved Brains & CSV Data", True, Color.ACCENT_RED)
        self.screen.blit(clear_t, (self.win_width//2 - clear_t.get_width()//2, self.win_height - 60))
        
        if self.notification_timer > 0:
            notif_font = pygame.font.SysFont(self.DEFAULT_FONT, 30, bold=True)
            notif = notif_font.render(self.notification_text, True, Color.ACCENT_MINT)
            self.screen.blit(notif, (self.win_width//2 - notif.get_width()//2, 30))
            self.notification_timer -= 1
            
        pygame.display.update()

    def generate_level(self, level):
        self.track.surface.fill(Color.GRASS_COLOR)
        self.stop_early = False
        self.abort_to_menu = False
        self.global_best_fitness = 0
        self.current_level_id = level
        
        def draw_track_path(pts, closed=False):
            pygame.draw.lines(self.track.surface, Color.TRACK_COLOR, closed, pts, 120)
            for p in pts:
                pygame.draw.circle(self.track.surface, Color.TRACK_COLOR, p, 60)
                
        if level in LEVELS:
            data = LEVELS[level]
            draw_track_path(data["pts"], closed=data["closed"])
            self.decided_car_pos = data["start"]
            self.end_point_pos = data["end"]
            self.state = "ai_running"

    def draw_floating_ui(self):
        font = pygame.font.SysFont(self.DEFAULT_FONT, 36, bold=True)
        text = ""
        if self.state == "drawing_track":
            text = "Step 1: Draw your road (SPACE to confirm)"
        elif self.state == "placing_finish_line":
            text = "Step 2: Click to place Finish Line (SPACE to skip)"
        elif self.state == "placing_start_point":
            text = "Step 3: Click to place Start Point"
            
        if text:
            rendered = font.render(text, True, Color.TEXT_DARK)
            bg = pygame.Surface((rendered.get_width() + 40, rendered.get_height() + 20), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 220))
            pygame.draw.rect(bg, Color.ACCENT_BLUE, bg.get_rect(), 3, border_radius=10)
            bg.blit(rendered, (20, 10))
            self.viewport.blit(bg, (self.VIEW_W//2 - bg.get_width()//2, 50))
            
        # Brush slider UI
        if self.brush_ui_timer > 0:
            self.brush_ui_timer -= 1
            slider_bg = pygame.Rect(self.VIEW_W//2 - 150, self.VIEW_H - 100, 300, 20)
            pygame.draw.rect(self.viewport, Color.PANEL_BG, slider_bg, border_radius=10)
            pygame.draw.rect(self.viewport, Color.ACCENT_BLUE, slider_bg, 2, border_radius=10)
            
            pct = (self.track.brush_size - self.track.MIN_BRUSH_SIZE) / (self.track.MAX_BRUSH_SIZE - self.track.MIN_BRUSH_SIZE)
            fill_rect = pygame.Rect(self.VIEW_W//2 - 150, self.VIEW_H - 100, 300 * pct, 20)
            pygame.draw.rect(self.viewport, Color.ACCENT_MINT, fill_rect, border_radius=10)
            
            f2 = pygame.font.SysFont(self.DEFAULT_FONT, 20)
            bt = f2.render(f"Brush Size: {self.track.brush_size}", True, Color.TEXT_DARK)
            self.viewport.blit(bt, (self.VIEW_W//2 - bt.get_width()//2, self.VIEW_H - 130))

    def render_viewport_to_screen(self):
        # Fill screen background
        self.screen.fill(Color.BG_LIGHT)
        
        # Scale viewport to fit screen while maintaining aspect ratio
        scale = min(self.win_width / self.VIEW_W, self.win_height / self.VIEW_H)
        new_w = int(self.VIEW_W * scale)
        new_h = int(self.VIEW_H * scale)
        
        scaled_viewport = pygame.transform.smoothscale(self.viewport, (new_w, new_h))
        offset_x = (self.win_width - new_w) // 2
        offset_y = (self.win_height - new_h) // 2
        
        # Draw shadow behind viewport
        shadow_rect = pygame.Rect(offset_x - 5, offset_y - 5, new_w + 10, new_h + 10)
        pygame.draw.rect(self.screen, (200, 205, 215), shadow_rect, border_radius=8)
        
        self.screen.blit(scaled_viewport, (offset_x, offset_y))
        pygame.draw.rect(self.screen, Color.ACCENT_BLUE, (offset_x, offset_y, new_w, new_h), 2)
        
        # Draw UI Panels around viewport
        font = pygame.font.SysFont(self.DEFAULT_FONT, 16)
        
        # Controls Panel (Top Right)
        controls_text = ["[L-Click] Draw / Start", "[R-Click] Erase", "[F] Drop Finish Line", "[S] Save / Stop", "[H] Hyper-Speed", "[V] Toggle Lasers", "[D] View Data", "[F5] Restart"]
        c_panel = pygame.Surface((200, 175))
        c_panel.fill(Color.PANEL_BG)
        pygame.draw.rect(c_panel, Color.ACCENT_BLUE, c_panel.get_rect(), 2)
        c_title = font.render("CONTROLS", True, Color.ACCENT_BLUE)
        c_panel.blit(c_title, (10, 10))
        for i, text in enumerate(controls_text):
            t = font.render(text, True, Color.TEXT_DARK)
            c_panel.blit(t, (10, 35 + i*17))
        self.screen.blit(c_panel, (self.win_width - 210, 10))
        
        # Stats Panel (Top Left)
        if self.state == "ai_running":
            s_panel = pygame.Surface((200, 80))
            s_panel.fill(Color.PANEL_BG)
            pygame.draw.rect(s_panel, Color.ACCENT_MINT, s_panel.get_rect(), 2)
            s_title = font.render("GLOBAL HIGH SCORE", True, Color.ACCENT_MINT)
            s_score = pygame.font.SysFont(self.DEFAULT_FONT, 30, bold=True).render(str(int(self.global_best_fitness)), True, Color.TEXT_DARK)
            s_panel.blit(s_title, (10, 10))
            s_panel.blit(s_score, (10, 35))
            self.screen.blit(s_panel, (10, 10))

        # Home Button
        if self.state != "main_menu":
            mx, my = pygame.mouse.get_pos()
            home_rect = pygame.Rect(10, 100 if self.state == "ai_running" else 10, 120, 40)
            bg_color = Color.ACCENT_RED if home_rect.collidepoint(mx, my) else Color.PANEL_BG
            text_color = Color.WHITE if home_rect.collidepoint(mx, my) else Color.TEXT_DARK
            
            pygame.draw.rect(self.screen, bg_color, home_rect, border_radius=8)
            pygame.draw.rect(self.screen, Color.ACCENT_RED, home_rect, 2, border_radius=8)
            
            h_text = font.render("<= HOME", True, text_color)
            self.screen.blit(h_text, (home_rect.x + home_rect.width//2 - h_text.get_width()//2, home_rect.y + 12))

        if self.show_data_viewer:
            self.draw_data_viewer()

        pygame.display.update()

    def draw_data_viewer(self):
        overlay = pygame.Surface((self.win_width, self.win_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150)) # dark background
        
        panel_w = 800
        panel_h = 600
        panel = pygame.Surface((panel_w, panel_h))
        panel.fill(Color.BG_LIGHT)
        pygame.draw.rect(panel, Color.ACCENT_BLUE, panel.get_rect(), 4, border_radius=15)
        
        font_title = pygame.font.SysFont(self.DEFAULT_FONT, 36, bold=True)
        font_head = pygame.font.SysFont(self.DEFAULT_FONT, 24, bold=True)
        font_body = pygame.font.SysFont(self.DEFAULT_FONT, 20)
        
        t = font_title.render("IN-GAME DATA VIEWER", True, Color.TEXT_DARK)
        panel.blit(t, (panel_w//2 - t.get_width()//2, 20))
        
        # Read Training Rounds
        y_offset = 80
        t2 = font_head.render("Recent Training Generations (csv_data/training_rounds.csv):", True, Color.ACCENT_BLUE)
        panel.blit(t2, (30, y_offset))
        y_offset += 40
        
        try:
            if os.path.exists("csv_data/training_rounds.csv"):
                with open("csv_data/training_rounds.csv", "r") as f:
                    lines = [l.strip() for l in f.readlines() if not l.startswith("#")]
                # Show last 8 lines
                panel.blit(font_body.render("Gen   |   Best Score   |   Surviving Cars", True, Color.TEXT_DARK), (40, y_offset))
                y_offset += 30
                for line in lines[-8:]:
                    cols = line.split(",")
                    row = f" {cols[0]:<10} | {cols[1]:<14} | {cols[2]}"
                    panel.blit(font_body.render(row, True, Color.TEXT_DARK), (40, y_offset))
                    y_offset += 25
            else:
                panel.blit(font_body.render("No training data generated yet.", True, Color.ACCENT_RED), (40, y_offset))
        except Exception as e:
            pass
            
        y_offset = 400
        t3 = font_head.render("Champion Car Stats (csv_data/champion_car.csv):", True, Color.ACCENT_MINT)
        panel.blit(t3, (30, y_offset))
        y_offset += 40
        try:
            if os.path.exists("csv_data/champion_car.csv"):
                with open("csv_data/champion_car.csv", "r") as f:
                    lines = [l.strip() for l in f.readlines() if not l.startswith("#")]
                if lines:
                    cols = lines[0].split(",")
                    panel.blit(font_body.render(f"Winning Generation: {cols[0]}", True, Color.TEXT_DARK), (40, y_offset))
                    panel.blit(font_body.render(f"Final Distance Score: {cols[1]}", True, Color.TEXT_DARK), (40, y_offset+30))
            else:
                panel.blit(font_body.render("No Champion Car saved yet. Press S to save!", True, Color.ACCENT_RED), (40, y_offset))
        except Exception:
            pass
            
        exit_t = font_body.render("Press [D] to close Data Viewer", True, Color.TEXT_DARK)
        panel.blit(exit_t, (panel_w//2 - exit_t.get_width()//2, panel_h - 40))
        
        overlay.blit(panel, (self.win_width//2 - panel_w//2, self.win_height//2 - panel_h//2))
        self.screen.blit(overlay, (0,0))

    def get_viewport_mouse_pos(self):
        mx, my = pygame.mouse.get_pos()
        scale = min(self.win_width / self.VIEW_W, self.win_height / self.VIEW_H)
        new_w = int(self.VIEW_W * scale)
        new_h = int(self.VIEW_H * scale)
        offset_x = (self.win_width - new_w) // 2
        offset_y = (self.win_height - new_h) // 2
        
        vx = (mx - offset_x) / scale
        vy = (my - offset_y) / scale
        return (int(vx), int(vy))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state != "main_menu":
                    mx, my = pygame.mouse.get_pos()
                    home_rect = pygame.Rect(10, 100 if self.state == "ai_running" else 10, 120, 40)
                    if home_rect.collidepoint(mx, my):
                        self.abort_to_menu = True
                        if self.state == "ai_running":
                            raise StopTraining()
                        self.state = "main_menu"
                        self.track.surface.fill(Color.GRASS_COLOR)
                        return True
                            
            if event.type == pygame.VIDEORESIZE:
                self.win_width, self.win_height = event.w, event.h
                self.screen = pygame.display.set_mode((self.win_width, self.win_height), pygame.RESIZABLE)
                
            if self.state == "main_menu":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = pygame.mouse.get_pos()
                    
                    cols = 4
                    btn_w = 320
                    btn_h = 50
                    gap_x = 20
                    gap_y = 15
                    total_w = cols * btn_w + (cols - 1) * gap_x
                    total_h = 4 * btn_h + 3 * gap_y
                    start_x = self.win_width // 2 - total_w // 2
                    start_y = self.win_height // 2 - total_h // 2 + 50
                    
                    for i in range(16):
                        row = i // cols
                        col = i % cols
                        x = start_x + col * (btn_w + gap_x)
                        y = start_y + row * (btn_h + gap_y)
                        rect = pygame.Rect(x, y, btn_w, btn_h)
                        
                        if rect.collidepoint(mx, my):
                            if i == 0:
                                self.state = "drawing_track"
                                self.track.surface.fill(Color.GRASS_COLOR)
                                self.ignore_drawing = True
                                self.stop_early = False
                                self.abort_to_menu = False
                                self.global_best_fitness = 0
                                self.current_level_id = 0
                            else:
                                self.generate_level(i)
                                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        try:
                            if os.path.exists("saved_brains"):
                                for f in os.listdir("saved_brains"):
                                    os.remove(os.path.join("saved_brains", f))
                            if os.path.exists("best_car.pkl"): os.remove("best_car.pkl")
                            self.notification_text = "Brains Wiped! (CSV Preserved)"
                            self.notification_timer = 180
                        except Exception:
                            pass
                            
                continue
            
            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                if self.state == "drawing_track":
                    self.state = "placing_finish_line"
                elif self.state == "placing_finish_line":
                    self.end_point_pos = None
                    self.state = "placing_start_point"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    if self.state == "placing_start_point":
                        self.state = "drawing_track"
                if event.key == pygame.K_s and self.state == "ai_running":
                    self.stop_early = True
                if event.key == pygame.K_p and self.state == "training_finished":
                    self.state = "playback"
                    self.start_playback()
                if event.key == pygame.K_f:
                    self.end_point_pos = self.get_viewport_mouse_pos()
                if event.key == pygame.K_h:
                    self.hyper_speed = not self.hyper_speed
                if event.key == pygame.K_v:
                    Car.DRAW_SENSORS = not Car.DRAW_SENSORS
                if event.key == pygame.K_d:
                    self.show_data_viewer = not self.show_data_viewer
                if event.key == pygame.K_UP:
                    self.track.adjust_brush_size(5)
                    self.brush_ui_timer = 90
                if event.key == pygame.K_DOWN:
                    self.track.adjust_brush_size(-5)
                    self.brush_ui_timer = 90
                if event.key == pygame.K_F5:
                    os.execl(sys.executable, sys.executable, *sys.argv)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.track.adjust_brush_size(1)
                    self.brush_ui_timer = 90
                elif event.button == 5:
                    self.track.adjust_brush_size(-1)
                    self.brush_ui_timer = 90

        return True
    
    def handle_drawing_track(self):
        if self.ignore_drawing:
            if not pygame.mouse.get_pressed()[0]:
                self.ignore_drawing = False
            return
            
        v_mouse = self.get_viewport_mouse_pos()
        if pygame.mouse.get_pressed()[0]:
            self.track.draw(v_mouse, Color.TRACK_COLOR)
        elif pygame.mouse.get_pressed()[2]:
            self.track.draw(v_mouse, Color.GRASS_COLOR)
        else:
            self.track.reset_last_position()

    def handle_placing_finish_line(self):
        v_mouse = self.get_viewport_mouse_pos()
        self.end_point_pos = list(v_mouse)
        
        if self.ignore_drawing:
            if not pygame.mouse.get_pressed()[0]:
                self.ignore_drawing = False
            return
            
        if pygame.mouse.get_pressed()[0]:
            self.state = "placing_start_point"
            self.ignore_drawing = True

    def handle_placing_start_point(self):
        v_mouse = self.get_viewport_mouse_pos()
        self.car.position = [
            v_mouse[0] - Car.CAR_SIZE_X / 2,
            v_mouse[1] - Car.CAR_SIZE_Y / 2
        ]
        
        if self.ignore_drawing:
            if not pygame.mouse.get_pressed()[0]:
                self.ignore_drawing = False
            return
            
        if pygame.mouse.get_pressed()[0]:
            self.decided_car_pos = self.car.position.copy()
            self.state = "ai_running"

    def draw(self):
        self.viewport.blit(self.track.get_surface(), (0, 0))
        
        # Draw Start Car
        if self.state in ["placing_start_point", "ai_running", "training_finished"]:
            self.viewport.blit(self.car.sprite, self.car.position)
            
        # Draw End Point
        if self.end_point_pos:
            pygame.draw.circle(self.viewport, Color.ACCENT_RED, self.end_point_pos, 40)
            pygame.draw.circle(self.viewport, Color.WHITE, self.end_point_pos, 30)
            pygame.draw.circle(self.viewport, Color.ACCENT_RED, self.end_point_pos, 20)
            pygame.draw.circle(self.viewport, Color.WHITE, self.end_point_pos, 10)
        
        if self.state == "training_finished":
            font1 = pygame.font.SysFont(self.DEFAULT_FONT, 32)
            font2 = pygame.font.SysFont(self.DEFAULT_FONT, 28)
            
            txt1 = font1.render("VICTORY! The AI reached the Finish Line.", True, Color.TEXT_DARK)
            txt2 = font2.render("Press P to Playback the Winning Car", True, Color.ACCENT_BLUE)
            
            popup_w = max(txt1.get_width(), txt2.get_width()) + 80
            popup_h = 200
            popup = pygame.Surface((popup_w, popup_h))
            popup.fill(Color.PANEL_BG)
            pygame.draw.rect(popup, Color.ACCENT_MINT, popup.get_rect(), 4)
            
            popup.blit(txt1, (popup_w//2 - txt1.get_width()//2, 60))
            popup.blit(txt2, (popup_w//2 - txt2.get_width()//2, 120))
            
            self.viewport.blit(popup, (self.VIEW_W//2 - popup_w//2, self.VIEW_H//2 - popup_h//2))
            
        self.draw_floating_ui()
        self.render_viewport_to_screen()

    def start_ai(self):
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.neat_config_path
        )
        self.neat_config = config
        population = neat.Population(config)

        if self.debug:
            population.add_reporter(neat.StdOutReporter(True))
            population.add_reporter(neat.StatisticsReporter())
            
        # Initialize the CSV Data Sheet only if it doesn't exist
        if not os.path.exists("csv_data/training_rounds.csv"):
            with open("csv_data/training_rounds.csv", "w") as f:
                f.write("# NEURODRIVE AI TRAINING DATA SHEET\n")
                f.write("# This sheet records how the AI improves over time.\n")
                f.write("# Generation: The current round of evolution.\n")
                f.write("# Best_Score: The furthest distance driven by the smartest car.\n")
                f.write("# Surviving_Cars: How many cars were still alive when the round ended.\n")
                f.write("# ---------------------------------------------------------\n")
                f.write("Generation,Best_Score,Surviving_Cars\n")

        try:
            best_genome = population.run(self.run_simulation, self.max_simulations)
        except StopTraining:
            if self.abort_to_menu:
                self.abort_to_menu = False
                self.state = "main_menu"
                self.track.surface.fill(Color.GRASS_COLOR)
                return
            best_genome = self.best_genome
        except Exception as e:
            # Catch NEAT extinction crashes or other errors safely
            self.notification_text = "AI Extinction Error! Returning to Menu."
            self.notification_timer = 240
            self.state = "main_menu"
            self.track.surface.fill(Color.GRASS_COLOR)
            return

        self.best_genome = best_genome
        
        filename = "saved_brains/custom_track_temp.pkl" if self.current_level_id == 0 else f"saved_brains/level_{self.current_level_id}_brain.pkl"
        with open(filename, "wb") as f:
            pickle.dump(best_genome, f)
        
        self.state = "training_finished"

    def run_simulation(self, genomes: List[neat.DefaultGenome], config: neat.Config) -> None:
        car_ai = CarAI(genomes, config, self.decided_car_pos, self.track)
        timer = time.time()

        while True:
            if not self.handle_events():
                exit(0)

            for idx, car in enumerate(car_ai.cars):
                if self.end_point_pos:
                    dist_to_finish = math.hypot(car.center[0] - self.end_point_pos[0], car.center[1] - self.end_point_pos[1])
                    if dist_to_finish <= 80:
                        self.stop_early = True
                        self.best_genome = genomes[idx][1]

            if self.stop_early:
                if not self.best_genome:
                    self.best_genome = genomes[0][1]
                    
                # Log to CSV before stopping
                with open("csv_data/training_rounds.csv", "a") as f:
                    f.write(f"{car_ai.TOTAL_GENERATIONS},{round(car_ai.best_fitness, 2)},{car_ai.remaining_cars}\n")
                    
                # Log Champion Car
                if not os.path.exists("csv_data/champion_car.csv"):
                    with open("csv_data/champion_car.csv", "w") as f:
                        f.write("# CHAMPION CAR STATS\n")
                        f.write("Winning_Generation,Final_Distance_Score\n")
                with open("csv_data/champion_car.csv", "a") as f:
                    f.write(f"{car_ai.TOTAL_GENERATIONS},{round(car_ai.best_fitness, 2)}\n")
                    
                raise StopTraining()

            car_ai.compute(self.track.get_surface())
            
            if car_ai.best_fitness > self.global_best_fitness:
                self.global_best_fitness = car_ai.best_fitness

            if car_ai.remaining_cars == 0 or time.time() - timer > CarAI.TIME_LIMIT:
                # Log generation stats to CSV at the end of the round
                with open("csv_data/training_rounds.csv", "a") as f:
                    f.write(f"{car_ai.TOTAL_GENERATIONS},{round(car_ai.best_fitness, 2)},{car_ai.remaining_cars}\n")
                break

            self.viewport.blit(self.track.get_surface(), (0, 0))
            for car in car_ai.cars:
                car.draw(self.viewport)

            if car_ai.best_nn:
                car_ai.best_nn.draw(self.viewport)
                
            if self.end_point_pos:
                pygame.draw.circle(self.viewport, Color.ACCENT_RED, self.end_point_pos, 40)
                pygame.draw.circle(self.viewport, Color.WHITE, self.end_point_pos, 30)
                pygame.draw.circle(self.viewport, Color.ACCENT_RED, self.end_point_pos, 20)
                pygame.draw.circle(self.viewport, Color.WHITE, self.end_point_pos, 10)

            caption = (f"Gen {car_ai.TOTAL_GENERATIONS} - Alive: {car_ai.remaining_cars} - Time: {round(CarAI.TIME_LIMIT - (time.time() - timer), 2)}s")
            pygame.display.set_caption(caption)

            self.draw_floating_ui()
            self.render_viewport_to_screen()
            
            fps_target = 0 if self.hyper_speed else self.FPS
            self.clock.tick(fps_target)

    def start_playback(self):
        self.car_ai = CarAI([(1, self.best_genome)], self.neat_config, self.decided_car_pos, self.track)

    def handle_playback(self):
        self.car_ai.compute(self.track.get_surface())
        if self.car_ai.remaining_cars == 0:
            self.start_playback()
            
        self.viewport.blit(self.track.get_surface(), (0, 0))
        for car in self.car_ai.cars:
            car.draw(self.viewport)
        if self.car_ai.best_nn:
            self.car_ai.best_nn.draw(self.viewport)
            
        if self.end_point_pos:
            pygame.draw.circle(self.viewport, Color.ACCENT_RED, self.end_point_pos, 40)
            pygame.draw.circle(self.viewport, Color.WHITE, self.end_point_pos, 30)
            pygame.draw.circle(self.viewport, Color.ACCENT_RED, self.end_point_pos, 20)
            pygame.draw.circle(self.viewport, Color.WHITE, self.end_point_pos, 10)
            
        self.render_viewport_to_screen()
        self.clock.tick(self.FPS)

    def run(self):
        while True:
            if not self.handle_events():
                break

            if self.state == "main_menu":
                self.draw_main_menu()
            elif self.state == "drawing_track":
                self.handle_drawing_track()
                self.draw()
            elif self.state == "placing_finish_line":
                self.handle_placing_finish_line()
                self.draw()
            elif self.state == "placing_start_point":
                self.handle_placing_start_point()
                self.draw()
            elif self.state == "ai_running":
                self.start_ai()
            elif self.state == "training_finished":
                self.draw()
            elif self.state == "playback":
                self.handle_playback()

            if self.state != "ai_running":
                self.clock.tick(self.FPS)

        pygame.quit()