J/A+A/618/A59 Gaia DR2 confirmed new nearby open clusters (Castro-Ginard+, 2018)
================================================================================
A new method for unveiling open clusters in Gaia.
New nearby open clusters confirmed by DR2.
    Castro-Ginard A., Jordi C., Luri X., Julbe F., Morvan M., Balaguer-Nunez L.,
    Cantat-Gaudin T.
   <Astron. Astrophys., 618, A59 (2018)>
   =2018A&A...618A..59C    (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Milky Way ; Galactic center ; Clusters, open
Keywords: surveys - open clusters and associations: general - astrometry -
          methods: data analysis

Abstract:
    The publication of the Gaia Data Release 2 (Gaia DR2) opens a new era
    in astronomy. It includes precise astrometric data (positions, proper
    motions, and parallaxes) for more than 1.3 billion sources, mostly
    stars. To analyse such a vast amount of new data, the use of
    data-mining techniques and machine-learning algorithms is mandatory.

    A great example of the application of such techniques and algorithms
    is the search for open clusters (OCs), groups of stars that were born
    and move together, located in the disc. Our aim is to develop a method
    to automatically explore the data space, requiring minimal manual
    intervention.

    We explore the performance of a density-based clustering algorithm,
    DBSCAN, to find clusters in the data together with a supervised
    learning method such as an artificial neural network (ANN) to
    automatically distinguish between real OCs and statistical clusters.

    The development and implementation of this method in a
    five-dimensional space (l, b, p, {mu}_{alpha}_^*^, {mu}_{delta}_) with
    the Tycho-Gaia Astrometric Solution (TGAS) data, and a posterior
    validation using Gaia DR2 data, lead to the proposal of a set of new
    nearby OCs.

    We have developed a method to find OCs in astrometric data, designed
    to be applied to the full Gaia DR2 archive.

Description:
    We have designed, implemented, and tested an automated datamining
    system for the detection of OCs using astrometric data. The method is
    based on i) DBSCAN, an unsupervised learning algorithm to find groups
    of stars in a N-dimensional space (our implementation uses five
    parameters l, b, varpi, pmRA*, pmDE) and ii) an ANN trained to
    distinguish between real OCs and spurious statistical clusters by
    analysis of CMDs. This system is designed to work with minimal manual
    intervention for its application to large datasets, and in particular
    to the Gaia second data release, Gaia DR2.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
centers.dat      230       23   Mean parameters for the reported UBC clusters
                                 (table2 of the paper)
members.dat      191     1318   Members for the reported UBC clusters
--------------------------------------------------------------------------------

See also:
   I/345 : Gaia DR2 (Gaia Collaboration, 2018)

Byte-by-byte Description of file: centers.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1-  7  A7    ---      Cluster   Cluster name (UBCNN_a) (UBC1-UBC32)
   9- 27 F19.15 deg      GLON      Galactic longitude mean
  29- 47 F19.17 deg    e_GLON      Galactic longitude standard deviation
  49- 68 F20.16 deg      GLAT      Galactic latitude mean
  70- 88 F19.17 deg    e_GLAT      Galactic latitude standard deviation
  90-107 F18.16 mas      Plx       Parallax mean
 109-128 F20.18 mas    e_Plx       Parallax standard deviation
 130-149 F20.17 mas/yr   pmRA*     Proper motion in right ascension mean
                                    (pmRA*cosDE)
 151-169 F19.17 mas/yr e_pmRA*     Proper motion in right ascension
                                    standard deviation
 171-190 F20.17 mas/yr   pmDE      Proper motion in declination mean
 192-210 F19.17 mas/yr e_pmDE      Proper motion in declination
                                    standard deviation
 212-217 F6.2   km/s     RV        ?=- Radial velocity
 219-223 F5.2   km/s   e_RV        ?=- rms uncertainty on RV
 225-227 I3     ---      N         Number of members found
 229-230 I2     ---      NRV       Number of members used to compute
                                    mean radial velocity
--------------------------------------------------------------------------------

Byte-by-byte Description of file: members.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  7  A7    ---     Cluster   Cluster name (UBCNN_a)
   9- 27  I19   ---     Source    Gaia DR2 source_id
  29- 47 F19.15 deg     GLON      Galactic longitude mean
  49- 68 F20.16 deg     GLAT      Galactic latitude mean
  70- 87 F18.16 mas     Plx       Parallax mean
  89-111 F23.19 mas/yr  pmRA*     Proper motion in right ascension (pmRA*cosDE)
 113-133 F21.18 mas/yr  pmDE      Proper motion in declination
 135-153 F19.16 mag     Gmag      Gaia G magnitude magnitude
 155-172 E18.13 mag     BPmag     ?=- Gaia BP magnitude
 174-191 E18.13 mag     RPmag     ?=- Gaia RP magnitude
--------------------------------------------------------------------------------

Acknowledgements:
    Alfred Castro-Ginard, acastro(at)fqa.ub.edu

================================================================================
(End) Alfred Castro-Ginard [Univ. Barcelona], Patricia Vannier [CDS] 06-Jun-2019
