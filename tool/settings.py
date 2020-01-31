env_setting = {
                "Boxoban-Train-v0":
                    {
                        "solved_reward": 0,
                        "update_timestep": 600,
                    }
               }

def get_env_setting(env_name):
    return env_setting[env_name]