J/A+A/661/A118  628 new open clusters found with OCfinder (Castro-Ginard+, 2022)
================================================================================
Hunting for open clusters in Gaia EDR3:
628 new open clusters found with OCfinder.
    Castro-Ginard A., Jordi C., Luri X., Cantat-Gaudin T., Carrasco, J.M.,
    Casamiquela L., Anders F., Balaguer-Nunez L., Badia R.M.
    <Astron. Astrophys. 661, A118 (2022)>
    =2022A&A...661A.118C        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Surveys ; Clusters, open ; Parallaxes, trigonometric ;
              Proper motions ; Radial velocities
Keywords: Galaxy: disc - open clusters and associations: general - astrometry -
          methods: data analysis

Abstract:
    The improvements in the precision of the published data in Gaia EDR3
    with respect to Gaia DR2, particularly for parallaxes and proper
    motions, offer the opportunity to increase the number of known open
    clusters in the Milky Way by detecting farther and fainter objects
    that have so far go unnoticed. Our aim is to keep completing the open
    cluster census in the Milky Way with the detection of new stellar
    groups in the Galactic disc.

    We use Gaia EDR3 up to magnitude G=18mag, increasing in one unit the
    magnitude limit and therefore the search volume explored in our previous
    studies.

    We use the OCfinder method to search for new open clusters in Gaia EDR3
    using a Big Data environment. As a first step, OCfinder identifies stellar
    statistical overdensities in the five dimensional astrometric space
    (position, parallax and proper motions) using the DBSCAN clustering
    algorithm. Then, these overdensities are classified into random statistical
    overdensities or real physical open clusters using a deep artificial neural
    network trained on well-characterised G, G_BP_-G_RP_ colour-magnitude
    diagrams.

    We report the discovery of 628 new open clusters within the Galactic
    disc, most of them located beyond 1 kpc from the Sun. From the
    estimation of ages, distances and line-of-sight extinctions of these
    open clusters, we see that young clusters align following the Galactic
    spiral arms while older ones are dispersed in the Galactic disc.
    Furthermore, we find that most open clusters are located at low
    Galactic altitudes with the exception of a few groups older than 1Gyr.

    We show the success of the OCfinder method leading to the discovery of
    a total of 1310 open clusters (joining the discoveries here with the
    previous ones based on Gaia DR2), which represents almost 50% of the
    know population. Our ability to perform big data searches on a large
    volume of the Galactic disc, together with the higher precision in
    Gaia EDR3, enable us to keep completing the census with the discovery
    of new open clusters.

Description:
    The methodology developed to search for new OCs in Gaia data,
    OCfinder, is described in detail in Paper I (Castro-Ginard et al.,
    2018A&A...618A..59C, Cat. J/A+A/618/A59. It was successfully applied
    to detect 23 new nearby OCs (Castro-Ginard et al.,2018A&A...618A..59C,
    Cat. J/A+A/618/A59) in the TGAS data set of Gaia DR1. It was also
    applied to Gaia DR2 where 53 new OCs were detected in a direction near
    the Galactic anticentre (Castro-Ginard et al. 2019) and hundreds of
    new OCs in a big data search on the whole Galactic disc (Castro-Ginard
    et al., 2020A&A...635A..45C, Cat. J/A+A/635/A45).

    Table 1 contains mean astrometric parameters for the detected
    clusters.

    Table 2 contains the member stars found for the new UBC open clusters
    reported in the paper. The columns are for astrometric parameters for
    the member stars, i.e. positions, parallax, proper motions and radial
    velocity when available, as well as the photometric information in the
    G, G_BP_ and G_RP_ bands. It also includes the Gaia source_id to allow
    the cross-match with other catalogues.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       133      628   Mean parameters for the reported UBC clusters
table2.dat       244    25466   Members for the reported UBC clusters
--------------------------------------------------------------------------------

See also:
         I/345 : Gaia DR2 (Gaia Collaboration, 2018)
 J/A+A/627/A35 : New open clusters in Galactic anti-centre (Castro-Ginard+ 2019)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1-  3  I3    ---      Seq       [0/627] cluster sequential number
   5- 11  A7    ---      Cluster   Cluster name (UBCNNN)
  13- 18  F6.2  deg      RAdeg     Right Ascension mean (ICRS) at Ep=2015.5
  20- 25  F6.2  deg    s_RAdeg     Right Ascension standard deviation
  27- 32  F6.2  deg      DEdeg     Declination mean (ICRS) at Ep=2015.5
  34- 37  F4.2  deg    s_DEdeg     Declination standard deviation
  39- 44  F6.2  deg      GLON      Galactic longitude mean
  46- 49  F4.2  deg    s_GLON      Galactic longitude standard deviation
  51- 56  F6.2  deg      GLAT      Galactic latitude mean
  58- 61  F4.2  deg    s_GLAT      Galactic latitude standard deviation
  63- 66  F4.2  mas      plx       Parallax mean
  68- 71  F4.2  mas    s_plx       Parallax standard deviation
  73- 77  F5.2  mas/yr   pmRA      Proper motion in right ascension mean,
                                    pmRA*cosDE
  79- 82  F4.2  mas/yr s_pmRA      Proper motion in right ascension standard
                                    deviation
  84- 88  F5.2  mas/yr   pmDE      Proper motion in declination mean
  90- 93  F4.2  mas/yr s_pmDE      Proper motion in declination standard
                                    deviation
  95-100  F6.2  km/s     RV        ? Radial velocity mean
 102-106  F5.2  km/s   s_RV        ? Radial velocity standard deviation
 108-110  I3    ---      Nmemb     Number of cluster members
     112  I1    ---      NmembRV   Number of cluster members with
                                    radial velocity measurements
     114  A1    ---      Flag      [a] Note (1)
 116-120  F5.3  [yr]     logAge    Logarithm of the cluster age
 122-127  F6.1  pc       Dist      Distance to the cluster
 129-133  F5.3  ---      AV        Line-of-sight extinction
--------------------------------------------------------------------------------
Note (1): Note as follows:
 a = positional coincidence with Kharchenko et al. (2013A&A...558A..53K,
      Cat. J/A+A/558/A53)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1-  5  I5    ---     Seq       [0/25465] Star sequential number
   7- 13  A7    ---     Cluster   Cluster name (UBCNNNN)
  15- 33  I19   ---     GaiaEDR3  Gaia EDR3 source_id
  35- 54 F20.16 deg     RAdeg     Right Ascension (ICRS) at Ep=2015.5
  56- 75 F20.16 deg     DEdeg     Declination (ICRS) at Ep=2015.5
  77- 96 F20.16 deg     GLON      Galactic longitude
  98-116 E19.16 deg     GLAT      Galactic latitude
 118-137 F20.16 mas     plx       Parallax
 139-159 E21.16 mas/yr  pmRA      Proper motion in right ascension, pmRA*cosDE
 161-182 E22.17 mas/yr  pmDE      Proper motion in declination
 184-203 F20.16 km/s    RV        ? Radial velocity
 205-222 F18.15 mag     Gmag      Gaia G magnitude
 224-244 F21.18 mag     BP-RP     ? Gaia BP-RP colour
--------------------------------------------------------------------------------

Acknowledgements:
    Alfred Castro-Ginard, acastro(at)strw.leidenuniv.nl

================================================================================
(End)    Alfred Castro-Ginard [Leiden Obs.], Patricia Vannier [CDS]  11-Apr-2022
