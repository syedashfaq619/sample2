/*
SQLyog Community Edition- MySQL GUI v6.07
Host - 5.0.27-community-nt : Database - drug
*********************************************************************
Server version : 5.0.27-community-nt
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

CREATE DATABASE IF NOT EXISTS `drug`;

USE `drug`;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

/*Table structure for table `t1` */

DROP TABLE IF EXISTS `t1`;

CREATE TABLE `t1` (
  `username` VARCHAR(50) DEFAULT NULL,
  `password` VARCHAR(50) DEFAULT NULL,
  `email` VARCHAR(50) DEFAULT NULL,
  `phoneno` VARCHAR(50) DEFAULT NULL
) ENGINE=INNODB DEFAULT CHARSET=latin1;

/*Data for the table `t1` */

INSERT  INTO `t1`(`username`,`password`,`email`,`phoneno`) VALUES ('a','b','c','d'),('hemant','hemant@123','hemant.loni4@gmail.com','9535957888'),('hemant','hemant@1323','hemant.loni4@gmail.com','9535957888'),('a','b','c','d'),('a','b','c','d');

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
