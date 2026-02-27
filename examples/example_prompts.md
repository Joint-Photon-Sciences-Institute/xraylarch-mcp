# Example Prompts for xraylarch-mcp

## Basic XANES

> Load the spectrum at ~/data/fe_foil.xdi and show me the normalized XANES.

> Normalize the data with e0=7112 eV and show the derivative.

> Compare the normalized XANES of fe_foil and fe2o3 on the same plot.

## EXAFS Analysis

> Extract the EXAFS from the iron foil data with rbkg=1.0 and plot chi(k)*k^2.

> Do a Fourier transform from k=2 to k=14 with a Kaiser window and show the R-space data.

> Show me the chi(R) magnitude plot with the first-shell peak labeled.

## Data Management

> What spectra do I have loaded?

> Show me the details of the fe_foil group - what arrays and processing has been done?

> Load all groups from the Athena project file at ~/data/iron_standards.prj.

## Advanced

> Merge the three iron foil scans (fe_foil_1, fe_foil_2, fe_foil_3) and normalize the result.

> Do a linear combination fit of unknown_1 using fe_metal, fe2o3, and feo as standards.

> Run PCA on all 10 loaded spectra and tell me how many components are significant.

> Smooth the normalized spectrum with sigma=2 and compare to the original.

## Custom Code

> Run this code in the larch session: `groups['fe_foil'].mu_corrected = groups['fe_foil'].mu * 1.05`
