using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandLogger : MonoBehaviour
{
    // OVRHand to track
    public OVRHand trackedHand;

    public GameObject jointMarkerPrefab;

    private bool logging_ = false;
    private string logHeader_ = "";
    private string logText_ = "";
    private List<OVRSkeleton.BoneId> loggedJoints;
    private List<OVRBone> loggedBones;
    private List<OVRSkeleton.BoneId> subloggedJoints;
    private System.DateTime logStartTime_;

    private bool initialized_ = false;


    iiscommon.Utilities.LogManager gLogManager;

    public struct JointSample
    {
        public Vector3 pos;
        public Quaternion rot;

        public JointSample(Vector3 p, Quaternion r)
        {
            this.pos = p;
            this.rot = r;
        }
    }
    // Start is called before the first frame update
    void Start()
    {
        loggedJoints = new List<OVRSkeleton.BoneId>();
        loggedBones = new List<OVRBone>();


        loggedJoints.Add(OVRSkeleton.BoneId.Hand_WristRoot); // TODO: need to adjust to palm
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_WristRoot);

        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Thumb0);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Thumb1);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Thumb2);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Thumb3);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_ThumbTip);

        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Index1);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Index2);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Index3);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_IndexTip);

        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Middle1);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Middle2);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Middle3);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_MiddleTip);

        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Ring1);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Ring2);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Ring3);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_RingTip);

        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Pinky0);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Pinky1);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Pinky2);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_Pinky3);
        loggedJoints.Add(OVRSkeleton.BoneId.Hand_PinkyTip);

        subloggedJoints.Add(OVRSkeleton.BoneId.Hand_WristRoot);
        subloggedJoints.Add(OVRSkeleton.BoneId.Hand_ThumbTip);
        subloggedJoints.Add(OVRSkeleton.BoneId.Hand_IndexTip);
        subloggedJoints.Add(OVRSkeleton.BoneId.Hand_MiddleTip);
        subloggedJoints.Add(OVRSkeleton.BoneId.Hand_RingTip);
        subloggedJoints.Add(OVRSkeleton.BoneId.Hand_PinkyTip);


        //loggedJoints.Add(TrackedHandJoint.Palm);
        //loggedJoints.Add(TrackedHandJoint.Wrist);
        //loggedJoints.Add(TrackedHandJoint.ThumbTip);
        //loggedJoints.Add(TrackedHandJoint.IndexTip);
        //loggedJoints.Add(TrackedHandJoint.MiddleTip);
        //loggedJoints.Add(TrackedHandJoint.RingTip);
        //loggedJoints.Add(TrackedHandJoint.PinkyTip);

        logHeader_ = string.Format("#cam_x,cam_y,cam_z,cam_qx,cam_qy,cam_qz,cam_qw," +
                                    "wrs_x,wrs_y,wrs_z,wrs_qx,wrs_qy,wrs_qz,wrs_qw," +
                                    "th0_x,th0_y,th0_z,th0_qx,th0_qy,th0_qz,th0_qw," +
                                    "th1_x,th1_y,th1_z,th1_qx,th1_qy,th1_qz,th1_qw," +
                                    "th2_x,th2_y,th2_z,th2_qx,th2_qy,th2_qz,th2_qw," +
                                    "th3_x,th3_y,th3_z,th3_qx,th3_qy,th3_qz,th3_qw," +
                                    "th4_x,th4_y,th4_z,th4_qx,th4_qy,th4_qz,th4_qw," +
                                    "in1_x,in1_y,in1_z,in1_qx,in1_qy,in1_qz,in1_qw," +
                                    "in2_x,in2_y,in2_z,in2_qx,in2_qy,in2_qz,in2_qw," +
                                    "in3_x,in3_y,in3_z,in3_qx,in3_qy,in3_qz,in3_qw," +
                                    "in4_x,in4_y,in4_z,in4_qx,in4_qy,in4_qz,in4_qw," +
                                    "mi1_x,mi1_y,mi1_z,mi1_qx,mi1_qy,mi1_qz,mi1_qw," +
                                    "mi2_x,mi2_y,mi2_z,mi2_qx,mi2_qy,mi2_qz,mi2_qw," +
                                    "mi3_x,mi3_y,mi3_z,mi3_qx,mi3_qy,mi3_qz,mi3_qw," +
                                    "mi4_x,mi4_y,mi4_z,mi4_qx,mi4_qy,mi4_qz,mi4_qw," +
                                    "ri1_x,ri1_y,ri1_z,ri1_qx,ri1_qy,ri1_qz,ri1_qw," +
                                    "ri2_x,ri2_y,ri2_z,ri2_qx,ri2_qy,ri2_qz,ri2_qw," +
                                    "ri3_x,ri3_y,ri3_z,ri3_qx,ri3_qy,ri3_qz,ri3_qw," +
                                    "ri4_x,ri4_y,ri4_z,ri4_qx,ri4_qy,ri4_qz,ri4_qw," +
                                    "pi0_x,pi0_y,pi0_z,pi0_qx,pi0_qy,pi0_qz,pi0_qw," +
                                    "pi1_x,pi1_y,pi1_z,pi1_qx,pi1_qy,pi1_qz,pi1_qw," +
                                    "pi2_x,pi2_y,pi2_z,pi2_qx,pi2_qy,pi2_qz,pi2_qw," +
                                    "pi3_x,pi3_y,pi3_z,pi3_qx,pi3_qy,pi3_qz,pi3_qw," +
                                    "pi4_x,pi4_y,pi4_z,pi4_qx,pi4_qy,pi4_qz,pi4_qw," +
                                    "timestamp,state");
    }

    // Update is called once per frame
    void Update()
    {
        if (!initialized_)
        {
            Initialize();
            return;
        }

        int boneCount = 0;
        foreach (OVRBone bone in loggedBones)
        {
            //if (boneCount == 0)
            //{
            //    if (transform.Find("bone-Palm") != null)
            //    {
            //        transform.Find("bone-Palm").position = bone.Transform.position;
            //        transform.Find("bone-Palm").rotation = bone.Transform.rotation;
            //        transform.Find("bone-Palm").Translate(0.06f, 0, 0, Space.Self);
            //    }                
            //}
            //else
            //{
                transform.Find("bone-" + bone.Id).position = bone.Transform.position;
            //}

            boneCount++;
        }

        if (logging_)
        {
            AppendTrackingData();
        }
    }

    public List<Vector3> HandJointsSample_()
    {
        List<Vector3> sample = new List<Vector3>();

        Transform camTransform = Camera.main.transform;
  

        int boneCount = 0;
        foreach (OVRBone bone in loggedBones)
        {

            Vector3 camRelPos = camTransform.InverseTransformPoint(bone.Transform.position);

            //if (boneCount == 0)
            //{
            //    if (transform.Find("bone-Palm") != null)
            //    {
            //        Vector3 palmRelPos = camTransform.InverseTransformPoint(transform.Find("bone-Palm").position);

            //        sample.Add(palmRelPos);                    
            //    }
            //}
            //else if (boneCount == 1)
            //{
            //    // Currently ignoring wrist joint
            //}
            //else
            //{               
                
            sample.Add(camRelPos);
            //}
            
            boneCount++;
        }

        return sample;
    }

    public List<JointSample> HandJointsSample()
    {
        List<JointSample> sample = new List<JointSample>();

        Transform camTransform = Camera.main.transform;

        int boneCount = 0;
        foreach (OVRBone bone in loggedBones)
        {
            Vector3 camRelPos = camTransform.InverseTransformPoint(bone.Transform.position);
            Quaternion camtRelRot = Quaternion.Inverse(camTransform.rotation) * bone.Transform.rotation;

            //if (bone.Id == OVRSkeleton.BoneId.Hand_WristRoot)
            //{
            //    //use absolute position and rotation for the wrist, while the rest use relative transform wrt wrist
            //    //wristRelPos = wristTransform.position; dont want wrist position, making it zero for now
            //    wristRelRot = wristTransform.rotation;
            //}
            // TODO, repeat for left hand                
            //if (boneCount == 0)
            //{
            //    if (transform.Find("bone-Palm") != null)
            //    {                                
            //        jointStr = string.Format("{0:F4},{1:F4},{2:F4},{3:F6},{4:F6},{5:F6},{6:F6},", transform.Find("bone-Palm").position.x, transform.Find("bone-Palm").position.y, transform.Find("bone-Palm").position.z, 0, 0, 0, 0);
            //    }
            //}
            //else
            //{
            //jointStr = string.Format("{0:F4},{1:F4},{2:F4},{3:F6},{4:F6},{5:F6},{6:F6},", bone.Transform.position.x, bone.Transform.position.y, bone.Transform.position.z, 0, 0, 0, 0);
            //jointStr = string.Format("{0:F4},{1:F4},{2:F4},{3:F6},{4:F6},{5:F6},{6:F6},", bone.Transform.position.x, bone.Transform.position.y, bone.Transform.position.z, bone.Transform.rotation.x, bone.Transform.rotation.y, bone.Transform.rotation.z, bone.Transform.rotation.w);

            JointSample joint = new JointSample(camRelPos, camtRelRot);
            //}
            sample.Add(joint);

            boneCount++;
        }

        return sample;
    }


    public bool Active()
    {
        return logging_;
    }
    
    public void LogStart()
    {
        Debug.Log("Start hand log");
        logging_ = true;

        // Set start time and clear log text
        logStartTime_ = System.DateTime.Now;
        logText_ = "";

        // Initialize logger
        gLogManager = new iiscommon.Utilities.LogManager();
        gLogManager.Initialize("gesture", false);

        // Write header
        gLogManager.WriteToLog(logHeader_);
    }

    public void LogStop()
    {
        Debug.Log("Stop hand log");
        logging_ = false;

        gLogManager.WriteToLog(logText_);
    }

    private void AppendTrackingData()
    {
        string line = "";

        // Get cam pose
        Vector3 camPos = Camera.main.transform.position;
        Quaternion camRot = Camera.main.transform.rotation;
        line = string.Format("{0:F4},{1:F4},{2:F4},{3:F6},{4:F6},{5:F6},{6:F6},", camPos.x, camPos.y, camPos.z, camRot.x, camRot.y, camRot.z, camRot.w);


        Transform camTransform = Camera.main.transform;

        int boneCount = 0;
        foreach (OVRBone bone in loggedBones)
        {
            Vector3 camRelPos = camTransform.InverseTransformPoint(bone.Transform.position);
            Quaternion camtRelRot = Quaternion.Inverse(camTransform.rotation) * bone.Transform.rotation;

            string jointStr = "0,0,0,0,0,0,-1,";

            // TODO, repeat for left hand                
            // TODO, currently no quaternion
            //if (boneCount == 0)
            //{
            //    if (transform.Find("bone-Palm") != null)
            //    {                                
            //        jointStr = string.Format("{0:F4},{1:F4},{2:F4},{3:F6},{4:F6},{5:F6},{6:F6},", transform.Find("bone-Palm").position.x, transform.Find("bone-Palm").position.y, transform.Find("bone-Palm").position.z, 0, 0, 0, 0);
            //    }
            //}
            //else
            //{
            jointStr = string.Format("{0:F4},{1:F4},{2:F4},{3:F6},{4:F6},{5:F6},{6:F6},", camRelPos.x, camRelPos.y, camRelPos.z, camtRelRot.x, camtRelRot.y, camtRelRot.z, camtRelRot.w);

            line += jointStr;

            boneCount++;
        }

        // Get timestamp
        System.DateTime currentTime = System.DateTime.Now;
        System.TimeSpan elapsedTime = currentTime - logStartTime_;
        string timestamp = Math.Round(elapsedTime.TotalMilliseconds).ToString();


        logText_ += line + timestamp + "\n";
    }

    private void Initialize()
    {
        OVRSkeleton skeleton = trackedHand.GetComponent<OVRSkeleton>();

        if (skeleton != null && skeleton.IsDataValid)
        {
            int boneCount = 0;

            foreach (OVRSkeleton.BoneId boneId in subloggedJoints)
            //foreach (OVRSkeleton.BoneId boneId in loggedJoints)
            {
                OVRBone bone = skeleton.Bones[(int)boneId];

                loggedBones.Add(bone);

                GameObject marker = Instantiate(jointMarkerPrefab, transform, true);
                marker.name = "bone-" + boneId;
                marker.transform.position = bone.Transform.position;

                //if (boneCount == 0)
                //{
                //    marker.name = "bone-Palm";
                //}

                boneCount++;
            }

            initialized_ = true;
        }
    }
}