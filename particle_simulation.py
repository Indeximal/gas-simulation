import numpy as np
import pygame


def calc_normal(arr):
    return np.array(arr) / np.linalg.norm(arr)


class Particle:
    def __init__(self, pos, vel, mass, radius):
        self.pos = np.array(pos).astype(float)
        self.vel = np.array(vel).astype(float)
        self.mass = float(mass)
        self.radius = float(radius)

    def tick(self, dt):
        self.pos += self.vel * dt

    def collide_with_surface(self, surface_normal):
        normal = calc_normal(surface_normal)
        if np.dot(self.vel, normal) < 0:
            new_vel = self.vel - 2 * np.dot(self.vel, normal) * normal
            self.vel = new_vel

    # Formula from https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
    # With explanation form https://stackoverflow.com/questions/35211114/2d-elastic-ball-collision-physics
    def collide_with_particle(self, other):
        print("Deprecated") 
        mass_scalar = (2 * other.mass) / (self.mass + other.mass)
        pos_diff = self.pos - other.pos
        dot_scalar = np.dot(self.vel - other.vel, pos_diff) / sum(pos_diff ** 2)
        new_vel = self.vel - mass_scalar * dot_scalar * pos_diff
        self.vel = new_vel

    def set_velocity(self, vel):
        self.vel = vel

    def draw(self, screen):
        r = int(self.radius)
        E = self.mass * np.linalg.norm(self.vel) * .001
        col_t = (-1 / (E + 1)) + 1
        c0 = np.array([0, 0, 0]) ** 2
        c1 = np.array([252, 185, 30]) ** 2
        c = tuple(np.sqrt(c1 * col_t + c0 * (1 - col_t)))
        p = tuple(self.pos.astype(int))
        pygame.draw.circle(screen, c, p, r, 0)


def update_screen():
    return pygame.display.set_mode(screen_size, pygame.RESIZABLE)

pygame.init()

screen_size = width, height = 500, 500
sizeArr = np.array(screen_size)

screen = update_screen()
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()

objects = [Particle((200, 200), (50, 0), 10, 15),
           Particle((300, 215), (-30, 0), 10, 15)]

bucket_count = 5, 5
physics_buckets = np.empty(bucket_count, dtype=object)

initial_speed = 100

running = True
simulating = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                new_particle = Particle(np.random.random(2) * np.array(screen_size),
                    (np.random.random(2) - 0.5) * 2 * initial_speed,
                    10,
                    15)
                objects.append(new_particle)
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_s:
                simulating = not simulating
        if event.type == pygame.VIDEORESIZE:
            screen_size = width, height = event.w, event.h
            update_screen()

    dt = clock.tick(100) / 1000.

    screen.fill((255, 255, 255)) # Draw background

    if simulating:
        # Do physics for every object
        for obj in objects:
            obj.tick(dt)

        for obj in objects:
            if obj.pos[0] < obj.radius:
                obj.collide_with_surface((1, 0))
            if obj.pos[0] >= width - obj.radius:
                obj.collide_with_surface((-1, 0))
            if obj.pos[1] < obj.radius:
                obj.collide_with_surface((0, 1))
            if obj.pos[1] >= height - obj.radius:
                obj.collide_with_surface((0, -1))

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                max_dist = (obj1.radius + obj2.radius) ** 2
                if sum((obj1.pos - obj2.pos) ** 2) < max_dist:
                    # Formula from https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
                    # With explanation form https://stackoverflow.com/questions/35211114/2d-elastic-ball-collision-physics
                    mass_scalar_1 = (2 * obj2.mass) / (obj1.mass + obj2.mass)
                    mass_scalar_2 = (2 * obj1.mass) / (obj1.mass + obj2.mass)
                    pos_diff = obj1.pos - obj2.pos
                    if np.dot(pos_diff, obj1.vel) > 0 and np.dot(pos_diff, obj2.vel) < 0:
                        continue
                    dot_scalar = np.dot(obj1.vel - obj2.vel, pos_diff) / sum(pos_diff ** 2)
                    vel_1 = obj1.vel - mass_scalar_1 * dot_scalar * pos_diff
                    vel_2 = obj2.vel + mass_scalar_1 * dot_scalar * pos_diff
                    obj1.set_velocity(vel_1)
                    obj2.set_velocity(vel_2)

    # Draw every objects
    for obj in objects:
        obj.draw(screen)

    pygame.display.flip() # Display frame


