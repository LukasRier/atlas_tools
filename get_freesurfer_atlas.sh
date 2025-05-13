#!/bin/bash
source load_FS
if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then 
		echo " "
		echo "Usage: $0 <FS_DIRECTORY> <subjectNumber> ['aal78'(default)|'glasser52']"
		echo " for running multiple atlases make the third argument \"aal78|glasser52\""
		echo "FS_DIRECTORY...directory containing freesurfer outputs for <subjectNumber>"
		echo "subjectNumber......For example sub-001, sub-002 etc."
		echo " "
		echo "Needs: Freesurfer, fsl,"
		echo "       <FS_DIRECTORY>/AAL_all_reg_1mm_FS.nii.gz (AAL region atlas - single 3D volume with region numbers in region voxels)"
		echo "			or "
		echo "       <FS_DIRECTORY>/Glasser52_all_reg_1mm_FS.nii.gz (Glasser 52 region atlas - single 3D volume with region numbers in region voxels)"
		echo "       <FS_DIRECTORY>/MNI152_T1_1mm_FS.nii.gz (MNI template)"
		echo " v2.0.0 | Lukas Rier 2025 "
		exit 0
fi 
export FSLOUTPUTTYPE=NIFTI_GZ
if [[ $# -ge "2" ]]
then
	FS_DIRECTORY=$1
	sub=$2
	atlas_choices=$3
else
	FS_DIRECTORY="$(zenity --file-selection \
                --title "Select Freesurfer Directory" \
				--directory \
                --filename="~")"

	subs="$(ls "$FS_DIRECTORY")"

	pipe_separated=$(printf "%s|" $subs)
	# Remove the trailing |
	pipe_separated=${pipe_separated%|}
	# echo $pipe_separated
	sub="$(zenity --forms --add-combo="Subject" --combo-values="$pipe_separated" \
				--title "Select Subject for Atlas alignment")"
	
	atlas_choices="$(zenity --list --checklist \
				--column="Select" --column="Atlas" \
				--title "Select Atlas" \
				TRUE "aal78" \
				FALSE "glasser52")"
fi

echo this is my atlas choice $atlas_choices

ATLAS_REGS_sub="${FS_DIRECTORY}/${sub}/mri/AAL_regions"
ATLAS_REGS="${FS_DIRECTORY}/AAL_all_reg_1mm_FS.nii.gz"

if [ -z "$atlas_choices" ]; then
        echo "Third argument is neither 'glasser52' nor 'aal78'."
		exit 1
fi

# Zenity returns a string separated by "|" if multiple selected
IFS='|' read -ra selected_atlases <<< "$atlas_choices"

ANAT="${FS_DIRECTORY}/${sub}/mri/brain"
mri_convert $ANAT.mgz $ANAT.nii.gz
MNI2ANAT="${FS_DIRECTORY}/${sub}/mri/mni2anat"
MNI_brain="${FS_DIRECTORY}/MNI152_T1_1mm_brain_FS.nii.gz"

for atlas_choice in "${selected_atlases[@]}";
do

	if [ "$atlas_choice" = "aal78" ]; then
		echo "using AAL78 atlas"
	elif [ "$atlas_choice" = "glasser52" ]; then
		echo "using Glasser52 atlas"
		ATLAS_REGS_sub="${FS_DIRECTORY}/${sub}/mri/Glasser52_regions"
		ATLAS_REGS="${FS_DIRECTORY}/Glasser52_all_reg_1mm_FS.nii.gz"
	fi

	# Check files exist
	if ! [[ -f "$ANAT.nii" || -f "$ANAT.nii.gz" ]]; then
		echo "$ANAT doesn't exist!"
		exit 1
	fi

	if ! [ -f "$ATLAS_REGS" ]; then
		echo "$ATLAS_REGS doesn't exist!"
		exit 1
	fi

	if ! [ -f "$MNI_brain" ]; then
		echo "$MNI_brain doesn't exist!"
		exit 1
	fi

	if [[ -f "$ATLAS_REGS_sub.nii" || -f "$ATLAS_REGS_sub.nii.gz" ]];then
		zenity --question \
			--title="Confirm Overwrite" \
			--text="$ATLAS_REGS_sub already exists. Do you want to overwrite?"
	fi

	if [ $? -ne 0 ]; then
		echo "Not overwriting $ATLAS_REGS_sub"
		continue
	fi

	# flirt anatomical to MNI saving ANAT2MNI transform
	echo "flirt ANAT -> MNI brain"
	flirt -in ${MNI_brain} -ref ${ANAT}.nii.gz -out ${MNI2ANAT} -omat ${MNI2ANAT}.mat \
	-bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear

	#Apply transform to AAL regions
	echo "warping AAL regions to subject space"
	flirt -in ${ATLAS_REGS} -applyxfm -init ${MNI2ANAT}.mat -out ${ATLAS_REGS_sub} -paddingsize 0.0 -interp nearestneighbour -ref ${ANAT}

	mri_convert ${ATLAS_REGS_sub}.nii.gz ${ATLAS_REGS_sub}.mgz
done
