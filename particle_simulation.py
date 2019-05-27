import numpy as np
import pygame


# Freestanding utils
def calc_normal(arr):
    return np.array(arr) / np.linalg.norm(arr)

def clamp_index(n, lenght):
    return max(min(n, lenght - 1), 0)

def is_valid_index(index, container):
    i_arr = np.array(index)
    shape = np.array(container.shape)
    return (i_arr < shape).all() and (i_arr >= 0).all()

def indices(x, y):
    return [(i, j) for i in range(x) for j in range(y)]


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

    def set_velocity(self, vel):
        self.vel = vel

    def draw(self, screen):
        r = int(self.radius)
        V = np.linalg.norm(self.vel)
        col_t = (-1 / (V + 1)) + 1
        c0 = np.array([0, 0, 0]) ** 2
        c1 = np.array([252, 185, 30]) ** 2
        c = tuple(np.sqrt(c1 * col_t + c0 * (1 - col_t)))
        p = tuple(self.pos.astype(int))
        pygame.draw.circle(screen, c, p, r, 0)


def update_screen():
    return pygame.display.set_mode(screen_size, pygame.RESIZABLE)


def get_speed_histogram(objects_list, bins=50, bin_size=30):
    counts = np.zeros(bins)
    for bucket in objects_list:
        for obj in bucket:
            V = np.linalg.norm(obj.vel)
            b = int(V / bin_size)
            if b < bins:
                counts[b] += 1
    return counts


def draw_histogram(screen, hist_data, max_height=100,
                   color=(200, 200, 200)):
    scale = max_height / max(hist_data)
    bins = len(hist_data)

    width = screen.get_width() / bins
    screen_height = screen.get_height()
    for i, height in enumerate(hist_data * scale):
        rect = (i * width, screen_height, width, -height)
        pygame.draw.rect(screen, color, rect, 0)


def random_helium(energy):
    helium_mass = 4
    helium_radius = 4
    v = np.sqrt(energy / helium_mass * 1000)
    angle = np.random.rand() * np.pi * 2
    vel = np.array([np.cos(angle), np.sin(angle)]) * v
    pos = np.random.random(2) * np.array(screen_size)
    return Particle(pos, vel, helium_mass, helium_radius)


def calc_bucket(obj, buckets_shape, screen_size):
    buckets_x, buckets_y = buckets_shape
    bucket_size_x = screen_size[0] / buckets_x
    bucket_size_y = screen_size[1] / buckets_y

    new_bucket_x = clamp_index(int(obj.pos[0] // bucket_size_x), buckets_x)
    new_bucket_y = clamp_index(int(obj.pos[1] // bucket_size_y), buckets_y)

    return (new_bucket_x, new_bucket_y)


def neighbor_buckets(index):
    i, j = index
    return [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]



# Init Pygame
pygame.init()
pygame.font.init()

screen_size = width, height = 900, 700
sizeArr = np.array(screen_size)

screen = update_screen()
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()

# Init Objects
bucket_count = buckets_x, buckets_y = 10, 10
physics_buckets = np.empty(bucket_count, dtype=list)
for i, j in indices(buckets_x, buckets_y):
    physics_buckets[i, j] = list()

for obj in [random_helium(e) for e in np.ones(200) * 1000]:
    b = calc_bucket(obj, bucket_count, screen_size)
    physics_buckets[b].append(obj)

# Init graphics
total_speed_hist = None
debug_font = pygame.font.Font(None, 24)

# Main Loop
running = True
simulating = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        # Key event
        if event.type == pygame.KEYDOWN:
            # New Particle
            if event.key == pygame.K_SPACE:
                new_particle = random_helium(np.random.rand() * 10)
                b = calc_bucket(new_particle, bucket_count, screen_size)
                physics_buckets[b].append(obj)
            # Quit
            if event.key == pygame.K_ESCAPE:
                running = False
            # Pause
            if event.key == pygame.K_s:
                simulating = not simulating
            # Reset Histogram
            if event.key == pygame.K_r:
                total_speed_hist = None
        # Resize
        if event.type == pygame.VIDEORESIZE:
            screen_size = width, height = event.w, event.h
            screen = update_screen()

    frame_time = clock.tick(100) / 1000.
    dt = 0.01

    if simulating:
        # Move every Particle
        for bucket in physics_buckets.flat:
            for obj in bucket:
                obj.tick(dt)

        # Update buckets
        for i, rows in enumerate(physics_buckets):
            for j, bucket in enumerate(rows):
                for obj in bucket:
                    b = calc_bucket(obj, bucket_count, screen_size)
                    if b != (i, j):
                        bucket.remove(obj)
                        physics_buckets[b].append(obj)

        # Particle-Wall collisions, optimized
        for bucket in physics_buckets[0]:
            for obj in bucket:
                if obj.pos[0] < obj.radius:
                    obj.collide_with_surface((1, 0))
        for bucket in physics_buckets[-1]:
            for obj in bucket:
                if obj.pos[0] >= width - obj.radius:
                    obj.collide_with_surface((-1, 0))
        for bucket in physics_buckets[..., 0]:
            for obj in bucket:
                if obj.pos[1] < obj.radius:
                    obj.collide_with_surface((0, 1))
        for bucket in physics_buckets[..., -1]:
            for obj in bucket:
                if obj.pos[1] >= height - obj.radius:
                    obj.collide_with_surface((0, -1))

        counter = 0
        # Particle-Particle collisions
        for bucket_index in indices(buckets_x, buckets_y):
            for other_index in neighbor_buckets(bucket_index):
                if not is_valid_index(other_index, physics_buckets):
                    continue
                for i, obj1 in enumerate(physics_buckets[bucket_index]):
                    # Prevents double calculation where obj1 and obj2 are swapped
                    start_index = i + 1 if bucket_index == other_index else 0
                    for obj2 in physics_buckets[other_index][start_index:]:
                        counter += 1
                        # Check collision with distance
                        max_dist = (obj1.radius + obj2.radius) ** 2
                        if sum((obj1.pos - obj2.pos) ** 2) > max_dist:
                            continue
                        # Formula from https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
                        # With explanation form https://stackoverflow.com/questions/35211114/2d-elastic-ball-collision-physics
                        mass_scalar_1 = (2 * obj2.mass) / (obj1.mass + obj2.mass)
                        mass_scalar_2 = (2 * obj1.mass) / (obj1.mass + obj2.mass)
                        pos_diff = obj1.pos - obj2.pos
                        # Prevents countinuos collisions by checking whether the two
                        # objects move towards each other.
                        if np.dot(pos_diff, obj1.vel) > 0 and np.dot(pos_diff, obj2.vel) < 0:
                            continue
                        dot_scalar = np.dot(obj1.vel - obj2.vel, pos_diff) / sum(pos_diff ** 2)
                        vel_1 = obj1.vel - mass_scalar_1 * dot_scalar * pos_diff
                        vel_2 = obj2.vel + mass_scalar_1 * dot_scalar * pos_diff
                        obj1.set_velocity(vel_1)
                        obj2.set_velocity(vel_2)

    #print(dt, counter)

    # Draw background
    screen.fill((255, 255, 255))

    # Draw histogram
    speed_hist = get_speed_histogram(physics_buckets.flat)
    if total_speed_hist is None:
        total_speed_hist = speed_hist
    else:
        total_speed_hist += speed_hist
    draw_histogram(screen, total_speed_hist)

    # Draw objects
    for bucket in physics_buckets.flat:
        for obj in bucket:
            obj.draw(screen)

    # Draw text
    simulation_speed = dt / frame_time
    text_surface = debug_font.render("sim @ {:.3f}x".format(simulation_speed), True, (0, 0, 0))
    screen.blit(text_surface, (10, 10))
    text_surface = debug_font.render("{} checks".format(counter), True, (0, 0, 0))
    screen.blit(text_surface, (10, 35))
    

    pygame.display.flip() # Display frame


