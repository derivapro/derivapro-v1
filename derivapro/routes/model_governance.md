## Model Governance

### Operational Controls

#### Overview

This test is to verify that the model document contains sufficient description of operational controls for the continued use of the model. Operational controls are required so that:

- Inputs to the model during production are complete and accurate
- Model users have sufficient levels of access to the model
- Model users have sufficient guidance to operate and troubleshoot the model
- There are sufficient backups and contingencies in place when the model’s operating system is no longer available.

As illustrated in the Model Documentation, the following operational control guidelines have been implemented in [name of client].

The key operating procedures and the user guide for the model are available from Murex. The Murex system has an online help feature that provides guidance to the users on operating the model. The model system has Graphical User Interfaces (GUI) for performing common end-user activities such as creating/loading trades, extracting market data, running the model, and viewing the PV/Risk reports. The online help can be accessed by pressing “F1” in a user session. This loads the Murex Doc Server containing the relevant documentation on Murex features and user guides.

#### Testing Performed

Model documentation was reviewed to assess the adequacy of operational controls including:

- Review of inputs while the model is in production
- User access controls
- Operating procedures for the model
- Model backup plans

##### Model Access and Security

Since the model is a third-party system, there is no direct access to the model source code. The primary user interaction with the model is mainly via the Murex user interface. At the time of go-live, different user groups (Front Office, Risk, Back Office) shall be given access to views and interfaces that are appropriate for their use of the model. This grant of access rights is controlled by the Front Office IT group, based on inputs received from representatives of the relevant model user groups.

##### Access Level Review

The Front Office IT group shall review the access levels on a periodic basis and seek approval for continued access to the Murex modules for various user groups. The access level review is governed by the policies and procedures for User Account Management.

##### Production Deployment

The production deployment process for this model is owned by the Front Office IT group and is governed by the policies and procedures for Deployment process as outlined in [reference].

##### Model Usage Controls

When the model goes live into production, automatic workflows are expected to be in place for loading relevant data and executing the model. In addition, many ad hoc calculations and scenarios may be executed by the user using various additional modules, based on the access rights available for the user. However, the official End of Day (EOD) feeds from the model to other downstream models are based on the official configurations agreed for the model.

##### Model Backup

In the event of model breakdown due to technical issues (such as market data issues, system failure, etc.), the Model Development team shall work with the model users to determine a suitable course of action. A variety of methods such as performing ad hoc calculations on a case-by-case basis or subjective overrides based on expert judgment shall be used. The specific course of action depends on the scope and severity of the model breakdown and shall be discussed with the respective users.


### Contingency Plans

[Name of Client] has implemented two contingency plans as illustrated below.

#### Disaster Recovery Plan

The Disaster Recovery (DR) plan for the platform/IT infrastructure supporting the model is governed by the IT Disaster Recovery Policies and Procedures.

#### Business Continuity Plan

The Business Continuity Plan (BCP) for the model is governed by the Third-Party Risk Management Framework. This is owned by the Operations Risk team and covers the BCP policies and procedures for all third-party and vendor models.
