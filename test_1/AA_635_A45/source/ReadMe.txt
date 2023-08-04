J/A+A/635/A45  570 new open clusters in the Galactic disc (Castro-Ginard+, 2020)
================================================================================
Hunting for open clusters in Gaia DR2: 582 new open clusters in the
Galactic disc.
    Castro-Ginard A., Jordi C., Luri X., Alvarez Cid-Fuentes J., Casamiquela L.,
    Anders F., Cantat-Gaudin T., Monguio M., Balaguer-Nunez L., Sola S.,
    Badia R.M.
    <Astron. Astrophys. 635, A45 (2020)>
    =2020A&A...635A..45C        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Surveys ; Clusters, open ; Parallaxes, trigonometric ;
              Proper motions ; Radial velocities
Keywords: surveys - open clusters and associations: general - astrometry -
          methods: data analysis

Abstract:
    Open clusters are key targets for studies of Galaxy structure and
    evolution, and stellar physics. Since the Gaia data release 2 (DR2),
    the discovery of undetected clusters has shown that previous surveys
    were incomplete.

    Our aim is to exploit the Big Data capabilities of machine learning to
    detect new open clusters in Gaia DR2, and to complete the open cluster
    sample to enable further studies of the Galactic disc.

    We use a machine-learning based methodology to systematically search
    the Galactic disc for overdensities in the astrometric space and
    identify the open clusters using photometric information. First, we
    used an unsupervised clustering algorithm, DBSCAN, to blindly search
    for these overdensities in Gaia DR2 (l, b, varpi, mu_alpha_*,
    mu_delta_), then we used a deep learning artificial neural network
    trained on colour-magnitude diagrams to identify isochrone patterns
    in these overdensities, and to confirm them as open clusters.

    We find 570 new open clusters distributed along the Galactic disc in
    the region |b|<20{deg}. We detect substructure in complex regions, and
    identify the tidal tails of a disrupting cluster UBC 274 of ~3Gyr
    located at ~2kpc.

    Adapting the mentioned methodology to a Big Data environment allows us
    to target the search using the physical properties of open clusters
    instead of being driven by computational limitations. This blind
    search for open clusters in the Galactic disc increases the number of
    known open clusters by 45%.

Description:
    Table 1 contains mean astrometric parameters for the detected
    clusters.

    Table 2 contains the member stars found for the new UBC open clusters
    reported in the paper. The columns are for astrometric parameters for
    the member stars, i.e. positions, parallax, proper motions and radial
    velocity when available, as well as the photometric information in the
    G, G_BP and G_RP bands. It also includes the Gaia source_id to allow
    the cross-match with other catalogues.
    Update: Total number of clusters is 570.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       310      570   Mean parameters for the reported UBC clusters
                                 (updated, 07-Apr-2020)
table2.dat       247    33635   Members for the reported UBC clusters
                                 (updated, 07-Apr-2020)
--------------------------------------------------------------------------------

See also:
         I/345 : Gaia DR2 (Gaia Collaboration, 2018)
 J/A+A/627/A35 : New open clusters in Galactic anti-centre (Castro-Ginard+ 2019)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1-  6  A6    ---      Cluster   Cluster name (UBCNNN)
   8- 27 F20.16 deg      RAdeg     Right Ascension mean (ICRS) at Ep=2015.5
  29- 50 F22.18 deg    e_RAdeg     Right Ascension standard deviation
  52- 72 F21.17 deg      DEdeg     Declination mean (ICRS) at Ep=2015.5
  74- 93 F20.18 deg    e_DEdeg     Declination standard deviation
  95-114 F20.16 deg      GLON      Galactic longitude mean
 116-135 F20.18 deg    e_GLON      Galactic longitude standard deviation
 137-158 F22.18 deg      GLAT      Galactic latitude mean
 160-179 F20.18 deg    e_GLAT      Galactic latitude standard deviation
 181-199 F19.17 mas      plx       Parallax mean
 201-221 F21.19 mas    e_plx       Parallax standard deviation
 223-244 F22.18 mas/yr   pmRA      Proper motion in right ascension mean,
                                    pmRA*cosDE
 246-265 F20.18 mas/yr e_pmRA      Proper motion in right ascension
                                    standard deviation
 267-287 E21.16 mas/yr   pmDE      Proper motion in declination mean
 289-308 F20.18 mas/yr e_pmDE      Proper motion in declination
                                    standard deviation
 310-310  A1    ---      Note      [ab] Note (1)
--------------------------------------------------------------------------------
Note (1): Note as follows:
 a = coincidence with Sim et al. (2019, J. Korean Astron. Soc., 52, 145) or
      Liu & Pang (2019ApJS..245...32L), see Sect. 4.1.5
 b = tentative identification with Kharchenko et al. (2013, Cat. J/A+A/558/A53),
      see Sect. 4.1.3
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 19  I19   ---     Source    Gaia source_id
  21- 41 E21.16 deg     RAdeg     Right Ascension (ICRS) at ep=2015.5
  43- 64 F22.18 deg     DEdeg     Declination (ICRS) at ep=2015.5
  66- 86 F21.17 deg     GLON      Galactic longitude
  88-110 F23.19 deg     GLAT      Galactic latitude
 112-130 F19.17 mas     plx       Parallax
 132-153 E22.16 mas/yr  pmRA      Proper motion in right ascension, pmRA*cosDE
 155-175 E21.16 mas/yr  pmDE      Proper motion in declination
 177-198 F22.17 km/s    RV        ? Radial velocity
 200-218 F19.16 mag     Gmag      Gaia G magnitude
 220-240 F21.18 mag     BP-RP     ? Gaia BP_RP
 242-247  A6    ---     Cluster   Cluster name
--------------------------------------------------------------------------------

Acknowledgements:
    Alfred Castro-Ginard, acastro(at)fqa.ub.edu

History:
    04-Mar-2020: on-line version
    03-Apr-2020: tables corrected (from author)
    07-Apr-2020: tables corrected (from author)

================================================================================
(End) Alfred Castro-Ginard [Univ. Barcelona], Patricia Vannier [CDS] 21-Jan-2020
