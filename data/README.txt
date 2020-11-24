Summary

	This document gives more detail on each of the 3 datasets included in this Fileset.
	For each dataset the following is given:
		- relevent papers that use the data or employ common techniques on similar data
		- more detailed description of the X features, and Y properties including
		  things to keep in mind when building ML models with the data.  The regression is 
		  expected to take the form of finding F in Y=F(X).

Dilute Solute Diffusion:

	Relevant papers:
		Original paper describing dataset in detail [1]:
		https://www.nature.com/articles/sdata201654
		Previous machine learning results [2]:
		https://www.sciencedirect.com/science/article/pii/S0927025617301738
	
	X features:
		The solute diffusion data includes two “material composition” columns which
		denote the elements used as the host and solute respectively. Material properties 
		have been generated from these host and solute elements and are given in columns
		four and onward. For example, the column "Site2_MeltingT" would be the solute 
		melting temperature for that host-solute pair. These properties have been generated 
		using the MAGPIE framework, and have been selected because they have been shown 
		in previous ML work to give good results when predicting these diffusion energy
		barriers [3].
		
	Y property:
		The material property this dataset seeks to predict is the effective diffusion
		activation energy for vacancy mediated diffusion of a dilute solute element
		in a known host crystal. The reported energies have all been normalized to the
		host value such that each energy in a host-solute pair is the relative energy
		to the respective host's self diffusion value. For example Al-Al could have
		an absolute value of 0.5 eV, but this value is subtracted off to bring the
		value in the dataset to 0 eV. Then every datapoint with Al as the host has the
		same adjustment made to it. If Al-Ag's absolute value is 0.7 eV, the reported
		value in the dataset is 0.2 eV (0.7 - 0.5). 
		
		There is also an alternative column with raw energy values instead of the normalized
		values. Using the raw values allows the particularly slow diffusers like W to be 
		separated out from the rest of the hosts.
	
Perovskite Stability:

	Relevant papers:
		Original paper with more dataset details and ML results [4]:
		https://www.sciencedirect.com/science/article/pii/S092702561830274X
		
	X features:
		The perovskite stability dataset includes columns giving detailed information about
		the materials composition and structure. An overall composition is given, and then
		that composition is broken down by which elements occupy each site within the
		perovskite structure. These columns were used to generate columns 12 and onward.
		The composition information in combination with the MAGPIE approach for
		generating elemental descriptors was used to generate these columns [3].
		For example, the "First Ionization Potential (V)_AB_avg" column gives the average
		value for atoms at the A and B sites in the perovskite structure.
		
	Y property:
		The perovskite Stability dataset includes the "energy_above_hull (meV/atom)"
		column as the property to be predicted. The energy above the convex hull gives
		an approximation for how stable each composition will be, and thus how easy
		the material may be to synthesize experimentally.

		A second property to predict is included in the “formation_energy (eV/atom)” column.
		The formation energy given is the DFT energy of the perovskite structure relative to the
		DFT energy of the appropriate concentrations of elemental end-members in their stable state.
		This property is not a direct measure of stability, but can still be an interesting property
		to predict.

Metallic Glass Descriptors:

	Relevant papers:
		There is no paper published on this data set at the present time.  The data was assembled
		primarily by Vanessa Nilsen under the guidance of Prof. Dane Morgan at UW Madison
		 (ddmorgan@wisc.edu).
		A previous study of reduced glass transition temperature as a GFA descriptor can be found in reference [5]:
		https://www.sciencedirect.com/science/article/pii/S0022309300000648
		
	X features:
		The metallic glass dataset gives two columns with information about the material
		Composition. The first is the overall composition, and the second is the highest
		Composition element. The columns from four to the end are the MAGPIE features that
		have been generated from the material composition column and give values such as
		properties averaged over the material composition as well as features that are only for 
		the majority element in each alloy [3]. The majority element features are labelled as 
		"site1".
		
	Y property:
		The reduced glass transition temperature (Trg) has historically been used as a rough 
		predictor for Glass Forming Ability (GFA). By making a model to predict Trg for an 
		arbitrary alloy, it could be possible to use these values to estimate GFA directly, or as 
		input for another model to then predict GFA.

Bibliography
[1]	H. Wu, T. Mayeshiba, D. Morgan, High-throughput ab-initio dilute solute diffusion database, Sci. Data. 3 (2016) 160054. doi:10.1038/sdata.2016.54.
[2]	H. Wu, A. Lorenson, B. Anderson, L. Witteman, H. Wu, B. Meredig, D. Morgan, Robust FCC solute diffusion predictions from ab-initio machine learning methods, Comput. Mater. Sci. 134 (2017) 160–165. doi:10.1016/J.COMMATSCI.2017.03.052.
[3]	L. Ward, A. Agrawal, A. Choudhary, C. Wolverton, A General-Purpose Machine Learning Framework for Predicting Properties of Inorganic Materials, Nat. Commun. (2015) 1–7. doi:10.1038/npjcompumats.2016.28.
[4]	W. Li, R. Jacobs, D. Morgan, Predicting the thermodynamic stability of perovskite oxides using machine learning models, Comput. Mater. Sci. 150 (2018) 454–463. doi:10.1016/j.commatsci.2018.04.033.
[5]	Z.P. Lu, Y. Li, S.C. Ng, Reduced glass transition temperature and glass forming ability of bulk glass forming alloys, J. Non. Cryst. Solids. 270 (2000) 103–114. doi:10.1016/S0022-3093(00)00064-8.

