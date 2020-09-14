def obs_intersect(obs_1, obs_2):
    intersect = True
    if obs_1[1] < obs_2[0] or \
        obs_2[1] < obs_1[0] or \
        obs_1[3] < obs_2[2] or \
        obs_2[3] < obs_1[2]:
        intersect = False
    return intersect

def is_free_state(x0, obstacles, posmin, posmax):
    """ Check if x0 is free state given list of obstacles"""
    if any([x0[0] >= obstacle[0] and x0[0] <= obstacle[1] and \
             x0[1] >= obstacle[2] and x0[1] <= obstacle[3] \
             for obstacle in obstacles]):
        return False
    return True

def find_obs(x0, n_obs, posmin, posmax, \
             border_size, box_buffer, min_box_size, max_box_size, \
             max_iter=100, ignore_intersection=True):
    """ Given state x0, place obstacles between x0 and posmax"""
    obs = []
    itr = 0
    while len(obs) < n_obs and itr < max_iter:
        xmin = (posmax[0] - border_size - max_box_size)*np.random.rand() + border_size
        xmin = np.max([xmin, x0[0]])
        xmax = xmin + min_box_size  + (max_box_size - min_box_size)*np.random.rand()
        ymin = (posmax[1] - border_size - max_box_size)*np.random.rand() + border_size
        ymin = np.max([ymin, x0[1]])
        ymax = ymin + min_box_size  + (max_box_size - min_box_size)*np.random.rand()
        obstacle = np.array([xmin - box_buffer, xmax + box_buffer, \
                        ymin - box_buffer, ymax + box_buffer])

        intersecting = False
        for obs_2 in obs:
            intersecting = obs_intersect(obstacle, obs_2)
            if intersecting:
                break
        if ignore_intersection:
            intersecting = False

        if is_free_state(x0, [obstacle], posmin, posmax) and not intersecting:
            obs.append(obstacle)
        itr += 1

    if len(obs) is not n_obs:
        return []
    return obs

def findIC(obstacles, posmin, posmax, velmin, velmax, max_iter=1000):
    """ Given list of obstacles, find IC that is collision free"""
    IC_found = False
    itr = 0
    while not IC_found and itr < max_iter:
        r0 = posmin + (posmax-posmin)*np.random.rand(2)
        if not any([r0[0] >= obstacle[0] and r0[0] <= obstacle[1] and \
             r0[1] >= obstacle[2] and r0[1] <= obstacle[3] \
             for obstacle in obstacles]):
            IC_found = True
    if not IC_found:
        return np.array([])
    x0 = np.hstack((r0, velmin + (velmax-velmin)*np.random.rand(2)))
    return x0

def random_obs(n_obs, posmin, posmax, border_size, box_buffer, \
              min_box_size, max_box_size, max_iter=100):
    """ Generate random list of obstacles in workspace """
    obstacles = []
    itr = 0
    while itr < max_iter and len(obstacles) is not n_obs:
        xmin = (posmax[0] - border_size - max_box_size)*np.random.rand() + border_size
        xmax = xmin + min_box_size  + (max_box_size - min_box_size)*np.random.rand()
        ymin = (posmax[1] - border_size - max_box_size)*np.random.rand() + border_size
        ymax = ymin + min_box_size  + (max_box_size - min_box_size)*np.random.rand()
        obstacle = np.array([xmin - box_buffer, xmax + box_buffer, \
                            ymin - box_buffer, ymax + box_buffer])
        intersecting = False
        for obs_2 in obstacles:
            intersecting = obs_intersect(obstacle, obs_2)
            if intersecting:
                break
        if not intersecting:
            obstacles.append(obstacle)
        itr += 1

    if len(obstacles) is not n_obs:
        obstacles = []
    return obstacles