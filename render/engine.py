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

class StopTraining(Exception): pass

# ------------------ CLASSES ------------------

class Engine:
    WIDTH = 1900
    HEIGHT = 950
    FPS = 60
    DEFAULT_FONT = "comicsansms"

    def __init__(self, neat_config_path: str, debug: bool, max_simulations: int):
        self.neat_config_path = neat_config_path
        self.debug = debug
        self.max_simulations = max_simulations
        self.title = "Neat Cars Race"
        
        pygame.init()
        pygame.display.set_caption(self.title)
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.track = Track(self.WIDTH, self.HEIGHT)
        self.car = Car([0, 0], self.track)
        self.decided_car_pos = None
        self.end_point_pos = None
        
        self.state = "drawing_track"
        self.instruction_index = 0
        
        self.stop_early = False
        self.best_genome = None
        self.neat_config = None
        
        self.hyper_speed = False
        self.global_best_fitness = 0

    def draw_instructions(self):
        # Creates a small, condensed control panel
        font = pygame.font.SysFont(self.DEFAULT_FONT, 18)
        panel = pygame.Surface((550, 120))
        panel.set_alpha(220)
        panel.fill((30, 30, 30))
        
        texts = [
            "--- CONTROLS ---",
            "[L-Click] Draw/Start   |   [R-Click] Erase   |   [UP/DOWN] Brush",
            "[SPACE] Confirm   |   [F] Finish Line   |   [S] Save Best Car",
            "[H] Hyper-Speed   |   [V] Lasers   |   [F5] Restart Program"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            rendered = font.render(text, True, color)
            panel.blit(rendered, (15, 10 + i * 25))
            
        pygame.draw.rect(panel, (0, 255, 255), panel.get_rect(), 2)
        
        # Position at bottom right with plenty of padding so it never clips
        self.screen.blit(panel, (self.WIDTH - 570, self.HEIGHT - 150))
        
        if self.state == "ai_running":
            score_panel = pygame.Surface((250, 50))
            score_panel.set_alpha(220)
            score_panel.fill((30, 30, 30))
            score_txt = font.render(f"HIGH SCORE: {int(self.global_best_fitness)}", True, (255, 255, 50))
            score_panel.blit(score_txt, (20, 10))
            pygame.draw.rect(score_panel, (255, 255, 50), score_panel.get_rect(), 2)
            self.screen.blit(score_panel, (self.WIDTH - 260, 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)
            
            if event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
                if self.state == "drawing_track":
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
                    self.end_point_pos = pygame.mouse.get_pos()
                if event.key == pygame.K_h:
                    self.hyper_speed = not self.hyper_speed
                if event.key == pygame.K_v:
                    Car.DRAW_SENSORS = not Car.DRAW_SENSORS
                if event.key == pygame.K_UP:
                    self.track.adjust_brush_size(5)
                if event.key == pygame.K_DOWN:
                    self.track.adjust_brush_size(-5)
                if event.key == pygame.K_F5:
                    os.execl(sys.executable, sys.executable, *sys.argv)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.track.adjust_brush_size(1)
                elif event.button == 5:
                    self.track.adjust_brush_size(-1)

        return True
    
    def handle_drawing_track(self):
        if pygame.mouse.get_pressed()[0]:
            self.track.draw(pygame.mouse.get_pos(), Color.BLACK)
        elif pygame.mouse.get_pressed()[2]:
            self.track.draw(pygame.mouse.get_pos(), Color.WHITE)
        else:
            self.track.reset_last_position()

    def handle_placing_start_point(self):
        if not pygame.mouse.get_pressed()[0]:
            self.car.position = [
                pygame.mouse.get_pos()[0] - Car.CAR_SIZE_X / 2,
                pygame.mouse.get_pos()[1] - Car.CAR_SIZE_Y / 2
            ]
        else:
            self.decided_car_pos = self.car.position.copy()
            self.state = "ai_running"

    def draw(self):
        self.screen.blit(self.track.get_surface(), (0, 0))
        
        # Draw Start Car
        if self.state in ["placing_start_point", "ai_running", "training_finished"]:
            self.screen.blit(self.car.sprite, self.car.position)
            
        # Draw End Point
        if self.state in ["placing_start_point", "ai_running", "training_finished", "playback"]:
            if self.end_point_pos:
                pygame.draw.circle(self.screen, (255, 50, 50), self.end_point_pos, 40)
                pygame.draw.circle(self.screen, (255, 255, 255), self.end_point_pos, 30)
                pygame.draw.circle(self.screen, (255, 50, 50), self.end_point_pos, 20)
                pygame.draw.circle(self.screen, (255, 255, 255), self.end_point_pos, 10)
        
        if self.state == "training_finished":
            font = pygame.font.SysFont(self.DEFAULT_FONT, 36)
            popup = pygame.Surface((650, 200))
            popup.fill((30, 30, 30))
            pygame.draw.rect(popup, (0, 255, 255), popup.get_rect(), 4)
            
            txt1 = font.render("VICTORY! The AI reached the Finish Line.", True, (255,255,255))
            txt2 = font.render("Press P to Playback the Winning Car", True, (0,255,0))
            
            popup.blit(txt1, (30, 50))
            popup.blit(txt2, (60, 120))
            
            self.screen.blit(popup, (self.WIDTH//2 - 325, self.HEIGHT//2 - 100))
            pygame.display.set_caption(f"{self.title} - Training Finished")
            
        self.draw_instructions()
        pygame.display.update()

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

        try:
            best_genome = population.run(self.run_simulation, self.max_simulations)
        except StopTraining:
            best_genome = self.best_genome

        self.best_genome = best_genome
        with open("best_car.pkl", "wb") as f:
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
                        car_ai.best_nn = car_ai.nns[idx]

            if self.stop_early:
                if car_ai.best_nn:
                    self.best_genome = car_ai.best_nn.genome
                else:
                    self.best_genome = genomes[0][1]
                raise StopTraining()

            car_ai.compute(self.track.get_surface())
            
            if car_ai.best_fitness > self.global_best_fitness:
                self.global_best_fitness = car_ai.best_fitness

            if car_ai.remaining_cars == 0 or time.time() - timer > CarAI.TIME_LIMIT:
                break

            self.screen.blit(self.track.get_surface(), (0, 0))
            for car in car_ai.cars:
                car.draw(self.screen)

            if car_ai.best_nn:
                car_ai.best_nn.draw(self.screen)
                
            if self.end_point_pos:
                pygame.draw.circle(self.screen, (255, 50, 50), self.end_point_pos, 40)
                pygame.draw.circle(self.screen, (255, 255, 255), self.end_point_pos, 30)
                pygame.draw.circle(self.screen, (255, 50, 50), self.end_point_pos, 20)
                pygame.draw.circle(self.screen, (255, 255, 255), self.end_point_pos, 10)

            caption = (f"Gen {car_ai.TOTAL_GENERATIONS} - "
                       f"Alive: {car_ai.remaining_cars} - "
                       f"Time Left: {round(CarAI.TIME_LIMIT - (time.time() - timer), 2)}s | "
                       f"High Score: {int(self.global_best_fitness)}")
            pygame.display.set_caption(caption)

            self.draw_instructions()
            pygame.display.update()
            
            fps_target = 0 if self.hyper_speed else self.FPS
            self.clock.tick(fps_target)

    def start_playback(self):
        self.car_ai = CarAI([(1, self.best_genome)], self.neat_config, self.decided_car_pos, self.track)

    def handle_playback(self):
        self.car_ai.compute(self.track.get_surface())
        
        if self.car_ai.remaining_cars == 0:
            self.start_playback()
            
        self.screen.blit(self.track.get_surface(), (0, 0))
        for car in self.car_ai.cars:
            car.draw(self.screen)
        if self.car_ai.best_nn:
            self.car_ai.best_nn.draw(self.screen)
            
        if self.end_point_pos:
            pygame.draw.circle(self.screen, (255, 50, 50), self.end_point_pos, 40)
            pygame.draw.circle(self.screen, (255, 255, 255), self.end_point_pos, 30)
            pygame.draw.circle(self.screen, (255, 50, 50), self.end_point_pos, 20)
            pygame.draw.circle(self.screen, (255, 255, 255), self.end_point_pos, 10)
            
        pygame.display.set_caption("Playback Mode - Winning AI! (Close window to exit)")
        
        font = pygame.font.SysFont(self.DEFAULT_FONT, 20)
        panel = pygame.Surface((300, 50))
        panel.set_alpha(220)
        panel.fill((30, 30, 30))
        pygame.draw.rect(panel, (0, 255, 0), panel.get_rect(), 2)
        txt = font.render("Close Window to Exit", True, (255,255,255))
        panel.blit(txt, (20, 10))
        self.screen.blit(panel, (10, 10))
        
        pygame.display.update()
        self.clock.tick(self.FPS)

    def run(self):
        while True:
            if not self.handle_events():
                break

            if self.state == "drawing_track":
                self.handle_drawing_track()
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