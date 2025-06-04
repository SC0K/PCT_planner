class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 30
    cost_barrier = 50
    coverage_threshold = 0.9


class ConfigWrapper():
    tomo_dir = '/rsc/tomogram/'
class ConfigSensor():
    sensor_range = 4
    sensor_fov = 100    # degrees
    sensor_fov_ver = 90
    sensor_fov_hor = 90    # degrees


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()
    sensor = ConfigSensor()