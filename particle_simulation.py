import numpy as np
import pygame


class Particle:
    def __init__(self, pos, vel):
        self.pos = np.array(pos).astype(float)
        self.vel = np.array(vel).astype(float)

    def tick(self, dt):
        self.pos += self.vel * dt

    def collide_with_surface(self, surface_normal):
        normal = np.array(surface_normal) / np.linalg.norm(surface_normal)
        if np.dot(self.vel, normal) < 0:
            new_vel = self.vel - 2 * np.dot(self.vel, normal) * normal
            self.vel = new_vel

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 0), tuple(self.pos.astype(int)), 5, 0)


def update_screen():
    return pygame.display.set_mode(screen_size, pygame.RESIZABLE)

pygame.init()

screen_size = width, height = 500, 500
sizeArr = np.array(screen_size)

screen = update_screen()
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()

objects = [Particle((200, 200), (93, 54))]

bucket_count = 5, 5
physics_buckets = np.empty(bucket_count, dtype=object)

initial_speed = 100

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                new_particle = Particle(np.random.random(2) * np.array(screen_size),
                    (np.random.random(2) - 0.5) * 2 * initial_speed)
                objects.append(new_particle)
            if event.key == pygame.K_ESCAPE:
                running = False
        if event.type == pygame.VIDEORESIZE:
            screen_size = width, height = event.w, event.h
            update_screen()

    dt = clock.tick() / 1000.

    screen.fill((255, 255, 255)) # Draw background

    # Do physics for every object
    for obj in objects:
        obj.tick(dt)

    for obj in objects:
        if obj.pos[0] < 0:
            obj.collide_with_surface((1, 0))
        if obj.pos[0] >= width:
            obj.collide_with_surface((-1, 0))
        if obj.pos[1] < 0:
            obj.collide_with_surface((0, 1))
        if obj.pos[1] >= height:
            obj.collide_with_surface((0, -1))

    # Draw every objects
    for obj in objects:
        obj.draw(screen)

    pygame.display.flip() # Display frame


