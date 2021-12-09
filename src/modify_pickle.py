import pickle

from experience_buffer import ExperienceBuffer

filename = input()
with open(filename, "rb") as p_f:
    experience, steps = pickle.load(p_f)

with open(filename, "wb") as p_f:
    pickle.dump([experience, 250000], p_f)
