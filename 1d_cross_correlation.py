import numpy as np

s = np.array([-1, 0, 0, 1, 1, 1, 0, -1, -1, 0, 1, 0, 0, -1])
t = np.array([1, 1, 0])
t_norm = np.sqrt(np.sum(pow(t, 2)))
out = []

print(s, t)

for i in range(len(s) - len(t) + 1):
	s_ = s[i:i+len(t)]
	#s_ = s_ - np.mean(s_)
	#z = np.sqrt(np.sum(pow(s_, 2)))

	#if z != 0:
	#	s_ = s_ / z

	#print()

	s_norm = np.sqrt(np.sum(pow(s_, 2)))
	cc_ = np.dot(s_ / s_norm, t / t_norm)



	c1 = np.dot(s_, t)
	c2 = np.sqrt(np.sum(pow(s_, 2)) * np.sum(pow(t, 2)))
	cc = c1 / c2

	print(cc, cc_)
	out.append(cc)

print(np.argmax(out))