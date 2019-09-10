## Purpose
<p>
This code has the intention to identify similar participants based on their movement. By movement, we use two different trip detection. A basic one with only dwell followed by trip every time a participant stays in the same place and move, respectively. And a more sofisticated method based on a kernel trip detection described by Benoit et al. [1]. Once the trips are detected, each trip sequence is converted to a text string which is applied with text mining methods to identify similar movement behaviors.
</p>

## Instructions
<p>
The main code can be found in victoria_analysis.py. In this folder, by default, we collect a subset of participants from the victoria database for test purposes. The text convertion is in the stable.py file while the basic trip detection code is in spatial_metrics.py and the kernel trip detection, in trip.detection.py.
</p>

## References
<p>
[1] Thierry, Benoit, Basile Chaix, and Yan Kestens. "Detecting activity locations from raw GPS data: a novel kernel-based algorithm." International journal of health geographics 12.1 (2013): 14.
</p>

#### Any question, email to luana.fragoso@usask.ca
